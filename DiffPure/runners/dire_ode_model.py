"""
ODE-based DIRE wrapper
----------------------

Purpose
- Provide a differentiable reconstruction pipeline for DIRE using the probability-flow ODE
  implemented in runners/diffpure_ode.py (VPODE + odeint_adjoint).
- Designed for white-box/e2e attacks: gradients propagate from DIRE map back to the input x0
  efficiently via adjoint method.

Contract
- Input: x0 in [-1, 1], shape [B, C, H, W]
- Output: a dict with keys:
  - recon: reconstructed x_hat0 (same shape as x0)
  - dire_map: per-pixel map in [0, 1] (abs-diff scaled), shape [B, C, H, W]
  - extras: { 't_scalar': float, 'eot_reps': int }

Key knobs
- t_steps (int): discrete noise steps (like args.t, 1..1000). Larger t means heavier noising before ODE reverse.
- method: ODE solver method ('euler', 'rk4', 'dopri5').
- rtol/atol: solver tolerances.
- step_size: for fixed-step solvers (e.g., euler, rk4).
- fix_rand (bool): use a fixed noise for stability in white-box attacks.
- eot_reps (int): number of EOT repetitions over the initial noise e. Results are averaged.

Notes
- This module avoids file I/O and does not depend on image saving from image_editing_sample.
- It reuses OdeGuidedDiffusion's internal model/VPODE and its discrete betas for consistent schedules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint

# Reuse existing ODE components and model loading
from .diffpure_ode import OdeGuidedDiffusion


@dataclass
class ODEDireConfig:
    # Discrete time step count in [1, N], where N is the training schedule length (typically 1000)
    t_steps: int = 50
    # ODE solver configs
    # Default to fixed-step Euler with a larger step size to match export/eval defaults
    method: Literal['euler', 'rk4', 'dopri5'] = 'euler'
    rtol: float = 1e-3
    atol: float = 1e-3
    step_size: float = 2e-2  # default fixed-step size aligned with export/eval
    # EOT and randomness
    fix_rand: bool = True
    eot_reps: int = 1
    # DIRE map shaping
    clamp_recon: bool = True  # clamp recon to [-1, 1]
    dire_scale_to_01: bool = True  # map |x_recon - x0| from [-2,2] to [0,1] by /2
    dire_smooth_eps: float = 5e-5  # epsilon for smoothing |.|; set 0 to recover exact abs
    dire_log_grad: bool = True  # default to False (can be enabled via args)
    # New stability / debug knobs
    force_fp32: bool = True  # force ODE forward in fp32 (disable external autocast influence)
    nan_fallback: bool = True  # if True, on NaN recon fallback to input (identity) and count event
    # Memory optimization toggles
    use_unet_checkpoint: bool = False  # enable UNet internal gradient checkpoint (if supported)
    disable_solver_fallback: bool = True  # if True, do not try secondary solvers (rk4/euler) to reduce duplicate graphs
    local_half: bool = False  # run vpode function inside autocast fp16 region (even if global amp off)
    # IMPORTANT: Freezing model params can break guided_diffusion's checkpoint/backward which
    # internally calls autograd.grad on params during adjoint VJP. Keep this False by default.
    freeze_model_params: bool = False  # set diffusion model params requires_grad_(False)
    empty_cache_each_iter: bool = True  # call torch.cuda.empty_cache() after each ODE solve (helps fragmentation)


class DIRE_ODE_Model(nn.Module):
    def __init__(
        self,
        args,  # existing argparse-like object used by OdeGuidedDiffusion for model loading
        config,  # training/config object used by OdeGuidedDiffusion
        ode_cfg: Optional[ODEDireConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.device = (
            device
            if device is not None
            else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        )

        # Initialize guided diffusion ODE runner (loads model, builds VPODE)
        self.ode_runner = OdeGuidedDiffusion(args=args, config=config, device=self.device)
        # Apply parameter freezing early if requested (not recommended with adjoint + checkpoint)
        if (ode_cfg and ode_cfg.freeze_model_params) or (not ode_cfg and ODEDireConfig.freeze_model_params):
            print("[dire-ode][warn] freeze_model_params=True may break adjoint gradients when guided_diffusion uses\n"
                  "checkpoint/backward with autograd.grad on params. Use with caution.")
            try:
                for p in self.ode_runner.model.parameters():
                    p.requires_grad_(False)
            except Exception:
                pass
        # Enable internal UNet checkpoint flag if desired (model exposes attribute in many repos)
        if ode_cfg and ode_cfg.use_unet_checkpoint:
            try:
                if hasattr(self.ode_runner.model, 'use_checkpoint'):
                    self.ode_runner.model.use_checkpoint = True
            except Exception:
                pass

        # Copy default solver configs from runner, allow overrides
        self.cfg = ode_cfg or ODEDireConfig()
        self.method = self.cfg.method
        self.rtol = self.cfg.rtol
        self.atol = self.cfg.atol
        self.step_size = self.cfg.step_size

        # Cached schedule (aligned with VPODE discrete betas)
        self.register_buffer('betas', self.ode_runner.betas)
        self.register_buffer('a_cum', (1.0 - self.betas).cumprod(dim=0))

        # Internal noise cache for fix_rand per input shape
        self._fixed_noise: Optional[torch.Tensor] = None
        # NaN event counter
        self.register_buffer('nan_events', torch.zeros(1, dtype=torch.long))
        self._last_good_recon: Optional[torch.Tensor] = None

    def _smooth_abs(self, diff_raw: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg.dire_smooth_eps)
        if eps > 0.0:
            eps_t = diff_raw.new_tensor(eps)
            return torch.sqrt(diff_raw.pow(2) + eps_t * eps_t)
        return diff_raw.abs()

    def _sample_xt(self, x0: torch.Tensor, t_steps: int, fix_rand: bool) -> torch.Tensor:
        """Forward noising to discrete step t: x_t = sqrt(a_t) x0 + sqrt(1 - a_t) e"""
        assert x0.dim() == 4, f"Expected BCHW, got {tuple(x0.shape)}"
        B, C, H, W = x0.shape

        t_steps = int(t_steps)
        if t_steps <= 0:
            return x0

        # Guard upper bound by schedule length
        max_steps = int(self.a_cum.shape[0])
        t_idx = min(t_steps, max_steps) - 1  # 0-based index

        a_t = self.a_cum[t_idx]
        sqrt_a = a_t.sqrt().to(x0)
        sqrt_1ma = (1.0 - a_t).sqrt().to(x0)

        if fix_rand:
            if (self._fixed_noise is None) or (self._fixed_noise.shape != x0.shape):
                # Create a per-instance fixed noise with a deterministic RNG, but without global seeding
                gen = torch.Generator(device=x0.device)
                gen.manual_seed(42)
                self._fixed_noise = torch.randn(
                    x0.shape,
                    generator=gen,
                    device=x0.device,
                    dtype=x0.dtype,
                )
            e = self._fixed_noise
        else:
            e = torch.randn_like(x0)

        x_t = sqrt_a * x0 + sqrt_1ma * e
        return x_t

    def _ode_reverse(self, x_t: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        """Solve probability-flow ODE from t0 -> t1 with adjoint, returning x_hat0.

        t0 is the continuous time in [0,1] corresponding to the discrete t_steps.
        A tiny positive t1 avoids singularity at 0.
        """
        B = x_t.shape[0]
        state = (x_t.view(B, -1),)
        ts = torch.linspace(t0, t1, 2, device=x_t.device)

        # Switch runner method if needed
        # We don't mutate the runner's method globally to avoid side effects
        vpode = self.ode_runner.vpode

        # Execute ODE solver
        fallback_chain = [(self.method, self.step_size)]
        if not self.cfg.disable_solver_fallback:
            if self.method not in ('rk4',):
                fallback_chain.append(('rk4', self.step_size))
            if self.method != 'euler':
                fallback_chain.append(('euler', max(self.step_size, 5e-4)))

        last_err: Optional[Exception] = None
        for method, step in fallback_chain:
            opts = None
            if method in ('euler', 'rk4'):
                opts = dict(step_size=step)
            try:
                # optional local half precision
                if self.cfg.local_half and torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        sol = odeint_adjoint(
                            vpode,
                            state,
                            ts,
                            atol=self.atol,
                            rtol=self.rtol,
                            method=method,
                            options=opts,
                        )
                else:
                    sol = odeint_adjoint(
                        vpode,
                        state,
                        ts,
                        atol=self.atol,
                        rtol=self.rtol,
                        method=method,
                        options=opts,
                    )
                break
            except AssertionError as err:
                last_err = err
                continue
            except RuntimeError as err:
                last_err = err
                continue
        else:
            raise RuntimeError("ODE solver failed for all fallbacks") from last_err

        x_hat0 = sol[0][-1].view_as(x_t)
        # Clamp reconstruction to valid range if enabled. (Bugfix: used wrong attr name _recon before.)
        if self.cfg.clamp_recon:
            x_hat0 = x_hat0.clamp(-1.0, 1.0)
        # NaN guard: if produced NaN/Inf revert to input or last good
        if self.cfg.nan_fallback and (torch.isnan(x_hat0).any() or torch.isinf(x_hat0).any()):
            self.nan_events += 1
            print(f"[dire-ode][warn] NaN/Inf in ODE recon (event #{int(self.nan_events.item())}); falling back to previous valid or x_t baseline.")
            if self._last_good_recon is not None and self._last_good_recon.shape == x_hat0.shape:
                x_hat0 = self._last_good_recon.detach()
            else:
                # fallback to noisy input projected to [-1,1]
                x_hat0 = x_t.clamp(-1.0, 1.0)
        else:
            self._last_good_recon = x_hat0.detach()
        if self.cfg.empty_cache_each_iter and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return x_hat0

    def _dire_map(self, x0: torch.Tensor, x_hat0: torch.Tensor) -> torch.Tensor:
        diff = self._smooth_abs(x_hat0 - x0)
        if self.cfg.dire_scale_to_01:
            diff = (diff / 2.0).clamp(0.0, 1.0)
        if self.cfg.dire_log_grad and x0.requires_grad:
            grad = torch.autograd.grad(diff.sum(), x0, retain_graph=True, allow_unused=True)
            grad_tensor = grad[0] if isinstance(grad, (tuple, list)) else grad
            if grad_tensor is None:
                print('[dire-ode] gradient unavailable (autograd returned None)')
            else:
                grad_detached = grad_tensor.detach()
                grad_norm = grad_detached.norm().item()
                grad_min = grad_detached.min().item()
                grad_max = grad_detached.max().item()
                print(f"[dire-ode] grad stats -> norm: {grad_norm:.6e}, min: {grad_min:.6e}, max: {grad_max:.6e}")
        return diff

    def reconstruct_once(self, x0: torch.Tensor, t_steps: Optional[int] = None, fix_rand: Optional[bool] = None) -> torch.Tensor:
        """Single reconstruction pass: x0 -> x_t -> x_hat0."""
        x0 = x0.to(self.device)
        t_steps = int(self.cfg.t_steps if t_steps is None else t_steps)
        fix_rand = self.cfg.fix_rand if fix_rand is None else fix_rand

        # Discrete -> continuous time mapping
        N = int(self.ode_runner.vpode.N)
        t0 = max(t_steps, 0) / float(N)
        t1 = 1e-5  # avoid t=0 singularity

        if self.cfg.force_fp32 and x0.dtype != torch.float32:
            x0 = x0.float()
        x_t = self._sample_xt(x0, t_steps=t_steps, fix_rand=fix_rand)
        # Always execute ODE solve in fp32 when force_fp32
        if self.cfg.force_fp32 and x_t.dtype != torch.float32:
            x_t_fp = x_t.float()
        else:
            x_t_fp = x_t
        x_hat0 = self._ode_reverse(x_t_fp, t0=t0, t1=t1)
        if self.cfg.force_fp32 and x_hat0.dtype != x0.dtype:
            # Cast back to original dtype (usually float32 anyway)
            pass
        return x_hat0

    def forward(self, x0: torch.Tensor, *, t_steps: Optional[int] = None, eot_reps: Optional[int] = None,
                fix_rand: Optional[bool] = None) -> Dict[str, Any]:
        """Compute ODE reconstruction and DIRE map with optional EOT averaging.

        Returns a dict with recon, dire_map, extras.
        """
        reps = int(self.cfg.eot_reps if eot_reps is None else eot_reps)
        fix = self.cfg.fix_rand if fix_rand is None else fix_rand
        t = int(self.cfg.t_steps if t_steps is None else t_steps)

        if reps <= 1:
            x_hat0 = self.reconstruct_once(x0, t_steps=t, fix_rand=fix)
            dire = self._dire_map(x0, x_hat0)
        else:
            # EOT: average over different noises; when fix_rand=True, force differing seeds
            # by perturbing the cached fixed noise slightly per rep (deterministic per idx)
            xs = []
            cache_backup = self._fixed_noise
            for i in range(reps):
                if fix:
                    # deterministically vary the fixed noise for each rep
                    gen = torch.Generator(device=x0.device)
                    gen.manual_seed(12345 + i)
                    self._fixed_noise = torch.randn(
                        x0.shape,
                        generator=gen,
                        device=x0.device,
                        dtype=x0.dtype,
                    )
                x_hat0_i = self.reconstruct_once(x0, t_steps=t, fix_rand=True if fix else False)
                xs.append(x_hat0_i)
            # restore fixed noise cache
            self._fixed_noise = cache_backup
            x_hat0 = torch.stack(xs, dim=0).mean(dim=0)
            dire = self._dire_map(x0, x_hat0)

        return {
            'recon': x_hat0,
            'dire_map': dire,
            'extras': {'t_scalar': float(t) / float(self.ode_runner.vpode.N), 'eot_reps': reps},
        }

    # --- Compatibility helpers to align with DDIM runner API ---
    def dire_map(self, x0: torch.Tensor, enable_grad: bool = True, normalize: bool = True) -> torch.Tensor:
        """Return DIRE map with a signature compatible to DDIM's dire_map.

        x0 is expected in [-1,1]. If normalize is True, return [0,1] map; otherwise return raw |recon-x0|.
        The enable_grad flag is kept for API compatibility.
        """
        out = self.forward(x0)
        if normalize:
            return out['dire_map']
        # recompute unnormalized diff to honor normalize=False
        diff = self._smooth_abs(out['recon'] - x0)
        if self.cfg.dire_log_grad and x0.requires_grad:
            grad = torch.autograd.grad(diff.sum(), x0, retain_graph=True, allow_unused=True)
            grad_tensor = grad[0] if isinstance(grad, (tuple, list)) else grad
            if grad_tensor is None:
                print('[dire-ode] gradient unavailable (autograd returned None)')
            else:
                grad_detached = grad_tensor.detach()
                grad_norm = grad_detached.norm().item()
                grad_min = grad_detached.min().item()
                grad_max = grad_detached.max().item()
                print(f"[dire-ode] grad stats -> norm: {grad_norm:.6e}, min: {grad_min:.6e}, max: {grad_max:.6e}")
        return diff

    def recon_from_x(self, x0: torch.Tensor, enable_grad: bool = True) -> torch.Tensor:
        """Return reconstruction in [-1,1], API-compatible with DDIM's recon_from_x."""
        return self.reconstruct_once(x0, t_steps=self.cfg.t_steps, fix_rand=self.cfg.fix_rand)
