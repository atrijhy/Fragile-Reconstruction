#!/usr/bin/env python3
"""
SD15-based ODE runner (latent space) with adjoint support.

This file is intentionally self-contained and modeled after
`runners/diffpure_ode.py`, but it uses a Stable Diffusion v1.5
checkpoint (diffusers format) as the diffusion backend.

Goal:
  - Provide a differentiable, memory-efficient reconstruction map
    x0 -> x_hat using odeint_adjoint in latent space.
  - Intended for research / debugging; not wired into DIRE by default.

Assumptions:
  - Local SD1.5 weights are available under a diffusers directory, e.g.
      /path/to/stable-diffusion-v1-5
    containing subfolders: vae, unet, text_encoder, tokenizer, scheduler.
    Set the SD15_MODEL_PATH environment variable or pass model_dir explicitly.

Basic usage (example):
  from runners.sd15_ode import SD15ODEConfig, SD15ODERunner
  cfg = SD15ODEConfig(
      model_dir="/path/to/stable-diffusion-v1-5",
      t_steps=20,
      method="dopri5",
      rtol=1e-3,
      atol=1e-3,
      fix_rand=True,
      prompt="a photo",
  )
  runner = SD15ODERunner(cfg, device="cuda:0")
  x = torch.rand(1, 3, 256, 256)  # in [0,1]
  x_hat = runner.reconstruct_from_x(x)

The reconstruction is defined as:
  x (image) -> latent z0 via VAE
           -> noisy latent z_T via DDPM forward at large timestep T
           -> integrate probability-flow ODE backward in continuous time
              from t=1 -> t=0 using odeint_adjoint
           -> decode z_hat0 via VAE to image space.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

try:
    # Prefer adjoint (constant memory) – this is the main point.
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:  # pragma: no cover
    from torchdiffeq import odeint  # type: ignore


@dataclass
class SD15ODEConfig:
    """Configuration for SD15 ODE runner."""

    model_dir: str
    # ODE integration settings
    t_steps: int = 20
    method: str = "dopri5"
    rtol: float = 1e-3
    atol: float = 1e-3
    # noise / randomness
    fix_rand: bool = True
    seed: int = 42
    # text conditioning
    prompt: str = "a photo"
    # image size (we will resize / center-crop to this)
    image_size: int = 256
    # whether to use fp16 inside UNet/VAE (saves memory but may be less stable)
    use_fp16: bool = False


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SD15VPFlow(nn.Module):
    """
    Continuous-time ODE for VP-SDE in latent space using SD1.5 UNet.

    We approximate the DDPM forward process as a VP-SDE:
      dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW
    and use the probability-flow ODE:
      dx/dt = -0.5 * beta(t) * x - beta(t) * eps_theta(x, t)

    Here:
      - `x` is the latent in SD1.5 latent space (scaled as in VAE).
      - `t` is normalized to [0, 1], where t=1 corresponds to the largest
        forward noise level (DDPM timestep T) and t=0 to the clean latent.

    This module is compatible with torchdiffeq.odeint[_adjoint].
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        text_cond: torch.Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.text_cond = text_cond  # [1, L, D]
        self.device = device

        alphas_cumprod = scheduler.alphas_cumprod.to(device)  # [T]
        self.num_train_timesteps = int(alphas_cumprod.shape[0])

        if hasattr(scheduler, "betas"):
            betas = scheduler.betas.to(device)
            self.beta_min = float(betas.min().item())
            self.beta_max = float(betas.max().item())
        else:
            # Reasonable fallback
            self.beta_min, self.beta_max = 0.1, 20.0

    def _t_to_index(self, t_cont: torch.Tensor) -> torch.Tensor:
        """Map continuous t∈[0,1] → fractional index [0, T-1]."""
        t_clamped = t_cont.clamp(0.0, 1.0)
        idx = t_clamped * (self.num_train_timesteps - 1)
        return idx

    def _predict_epsilon(self, x: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
        """
        Predict epsilon via UNet.

        Args:
            x: [B, C, H, W] latent
            t_cont: [B] or scalar, normalized time in [0,1]
        """
        idx = self._t_to_index(t_cont)
        idx_int = idx.round().to(torch.long).clamp(0, self.num_train_timesteps - 1)
        timesteps = idx_int.to(self.device)  # [B]

        # Broadcast text_cond if B>1
        if self.text_cond.shape[0] == 1 and x.size(0) > 1:
            cond = self.text_cond.expand(x.size(0), -1, -1)
        else:
            cond = self.text_cond

        out = self.unet(x, timesteps, encoder_hidden_states=cond).sample
        return out

    def forward(self, t: torch.Tensor, states: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        ODE RHS: dx/dt = -0.5 * beta(t) * x - beta(t) * eps_theta(x,t).
        """
        x = states[0]  # [B, C, H, W]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)

        if t.dim() == 0:
            t_batch = t.expand(x.size(0))
        else:
            t_batch = t.to(self.device).view(-1).expand(x.size(0))

        beta_t = self.beta_min + (self.beta_max - self.beta_min) * t_batch
        beta_t = beta_t.view(-1, 1, 1, 1).to(self.device)  # [B,1,1,1]

        eps = self._predict_epsilon(x, t_batch)  # [B,C,H,W]

        dx_dt = -0.5 * beta_t * x - beta_t * eps
        return (dx_dt,)


class SD15ODERunner:
    """
    SD1.5-based ODE reconstruction runner.

    This class:
      - loads SD1.5 (VAE + UNet + TextEncoder + Scheduler) from a local dir;
      - provides `reconstruct_from_x(x)` mapping an image in [0,1] to a
        reconstructed image in [0,1] via:
          x -> z0 -> z_T -> ODE (t=1→0) -> z_hat0 -> x_hat
      - uses odeint_adjoint internally to keep memory usage roughly constant
        w.r.t. the number of solver steps `t_steps`.
    """

    def __init__(self, cfg: SD15ODEConfig, device: str | torch.device = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        _set_seed(cfg.seed)

        # Load SD1.5 components from local diffusers directory.
        print(f"[SD15ODERunner] loading models from: {cfg.model_dir}")
        self.vae = AutoencoderKL.from_pretrained(cfg.model_dir, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(cfg.model_dir, subfolder="unet")
        self.tokenizer = CLIPTextModel.from_pretrained(cfg.model_dir, subfolder="text_encoder")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model_dir, subfolder="text_encoder")
        self.scheduler = DDIMScheduler.from_pretrained(cfg.model_dir, subfolder="scheduler")

        if cfg.use_fp16:
            self.vae = self.vae.half()
            self.unet = self.unet.half()
            self.text_encoder = self.text_encoder.half()

        self.vae.to(self.device).eval()
        self.unet.to(self.device).eval()
        self.text_encoder.to(self.device).eval()

        for m in (self.vae, self.unet, self.text_encoder):
            for mod in m.modules():
                if isinstance(mod, nn.Dropout):
                    mod.p = 0.0

        # Precompute text condition
        prompt = cfg.prompt
        tok = CLIPTokenizer.from_pretrained(cfg.model_dir, subfolder="tokenizer")
        inputs = tok(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=tok.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            text_emb = self.text_encoder(**inputs).last_hidden_state  # [1,L,D]
        self.text_cond = text_emb.detach()

    @torch.no_grad()
    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image in [0,1] to SD1.5 latent.
        x: [B,3,H,W] in [0,1]
        """
        # SD expects [-1,1]
        img = (x * 2.0 - 1.0).to(self.device)
        posterior = self.vae.encode(img).latent_dist
        lat = posterior.sample() * self.vae.config.scaling_factor
        return lat

    @torch.no_grad()
    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image in [0,1].
        z: [B,4,H',W']
        """
        z_scaled = z / self.vae.config.scaling_factor
        img = self.vae.decode(z_scaled).sample
        img = (img / 2.0 + 0.5).clamp(0.0, 1.0)
        return img

    def reconstruct_from_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic reconstruction mapping using ODE inversion.

        x: [B,3,H,W] in [0,1]
        Returns:
          x_hat: [B,3,H,W] in [0,1]
        """
        cfg = self.cfg
        B = x.size(0)

        # Encode image to latent z0
        with torch.no_grad():
            z0 = self._encode_image(x)  # [B,4,H',W']

        # Choose largest scheduler timestep as "T"
        T_idx = int(self.scheduler.num_train_timesteps) - 1
        alpha_T = self.scheduler.alphas_cumprod[T_idx].to(self.device)
        sqrt_alpha = alpha_T.sqrt()
        sqrt_1m_alpha = (1.0 - alpha_T).sqrt()

        # Fixed noise for determinism if requested
        if cfg.fix_rand:
            _set_seed(cfg.seed)
        noise = torch.randn_like(z0, device=self.device)

        # Forward DDPM to z_T
        z_T = sqrt_alpha * z0 + sqrt_1m_alpha * noise

        # Build ODE module
        ode_module = SD15VPFlow(
            unet=self.unet,
            scheduler=self.scheduler,
            text_cond=self.text_cond,
            device=self.device,
        ).to(self.device)

        # Integrate from t=1 -> t=0
        t0 = torch.tensor(1.0, device=self.device)
        t1 = torch.tensor(0.0, device=self.device)
        ts = torch.linspace(t0, t1, steps=cfg.t_steps, device=self.device)

        z_init = z_T.detach()  # we usually do not want grad wrt noise here
        states = (z_init,)

        z_traj = odeint(
            ode_module,
            states,
            ts,
            atol=cfg.atol,
            rtol=cfg.rtol,
            method=cfg.method,
        )[0]  # [len(ts),B,C,H,W]

        z_hat0 = z_traj[-1]

        # Decode back to image space
        with torch.no_grad():
            x_hat = self._decode_latent(z_hat0)
        return x_hat

