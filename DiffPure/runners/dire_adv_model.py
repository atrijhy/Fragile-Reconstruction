# dire_adv_model_with_autograd_bridge.py
import time
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import autograd

from runners.diffpure_guided_ddim import GuidedDiffusionDDIM
from runners.dire_ode_model import DIRE_ODE_Model, ODEDireConfig
from utils import get_image_classifier

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Attempt to import repository-provided set_seed; fallback to local implementation.
try:
    # prefer project's seed util if available
    from DIRE.utils.seed import set_seed as _repo_set_seed  # type: ignore
    def _set_seed_fixed():
        _repo_set_seed(42)
except Exception:
    def _set_seed_fixed():
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        os.environ['PYTHONHASHSEED'] = str(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

# Helper to try enabling PyTorch deterministic algorithms (optional; may raise)
def _try_enable_torch_deterministic():
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # older torch or not supported; ignore
        pass


class _LinearizedLogitsFunction(autograd.Function):
    """
    Custom autograd.Function:
      forward: compute logits = classifier(cached_y_for_cls)
      backward: compute grad_x via cached graph:
        v = dLoss/d(cached_y_for_cls)
        grad_y = (d cached_y_for_cls / d cached_y)^T * v
        grad_x = (d cached_y / d cached_x)^T * grad_y
    """

    @staticmethod
    def forward(ctx, x_cur, cached_y_for_cls, cached_y, cached_x, classifier):
        # Save tensors for backward
        # Keep them part of the graph (no detach) so autograd.grad can trace
        ctx.save_for_backward(cached_y_for_cls, cached_y, cached_x)
        ctx.classifier = classifier
        logits = classifier(cached_y_for_cls)
        return logits

    @staticmethod
    def backward(ctx, grad_output):
        cached_y_for_cls, cached_y, cached_x = ctx.saved_tensors
        classifier = ctx.classifier

        # v = dLoss / d(cached_y_for_cls)
        logits_y = classifier(cached_y_for_cls)
        v = autograd.grad(outputs=logits_y, inputs=cached_y_for_cls,
                          grad_outputs=grad_output, retain_graph=True, create_graph=False)[0]

        # grad_y = (d cached_y_for_cls / d cached_y)^T * v
        grad_y = autograd.grad(outputs=cached_y_for_cls, inputs=cached_y,
                               grad_outputs=v, retain_graph=True, create_graph=False)[0]

        # grad_x = (d cached_y / d cached_x)^T * grad_y
        grad_x = autograd.grad(outputs=cached_y, inputs=cached_x,
                               grad_outputs=grad_y, retain_graph=False, create_graph=False)[0]

        return grad_x, None, None, None, None


class DIRE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        """
        args: attack/runtime args (must include classifier_name, dire_backend, domain, etc.)
        config: global config (may contain device)
        This constructor will:
          - fix RNGs to seed=42 for reproducibility (best-effort)
          - build recon backend (DDIM or ODE) and classifier
          - try to put recon internal model into eval() for consistent inference
        """
        super().__init__()
        self.args = args
        self.config = config
        self.device = getattr(config, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # ----------------- MAKE RNG/DETERMINISM BEST-EFFORT -----------------
        # Always fix seed=42 here (explicit user request)
        try:
            _set_seed_fixed()
        except Exception:
            try:
                _set_seed_fixed()
            except Exception:
                pass
        _try_enable_torch_deterministic()
        # ------------------------------------------------------------------

        # classifier
        self.classifier = get_image_classifier(
            args.classifier_name,
            ckpt_path=getattr(args, 'classifier_ckpt', None)
        ).to(self.device)
        try:
            self.classifier.eval()
        except Exception:
            pass

        # diffusion backend
        self.backend = getattr(args, 'dire_backend', 'ddim')
        if self.backend == 'ddim':
            self.recon_model = GuidedDiffusionDDIM(args, config, device=self.device)
        elif self.backend == 'ode':
            ode_cfg = ODEDireConfig(
                t_steps=getattr(args, 't', 50),
                method=getattr(args, 'ode_method', 'euler'),
                rtol=getattr(args, 'ode_rtol', 1e-3),
                atol=getattr(args, 'ode_atol', 1e-3),
                step_size=getattr(args, 'ode_step_size', 2e-2),
                fix_rand=getattr(args, 'fix_rand', True),
                eot_reps=getattr(args, 'eot_reps', 1),
                clamp_recon=True,
                dire_scale_to_01=True,
                dire_smooth_eps=float(getattr(args, 'ode_dire_smooth_eps', 5e-5)),
                dire_log_grad=bool(getattr(args, 'ode_dire_log_grad', False)),
                force_fp32=bool(getattr(args, 'ode_force_fp32', True)),
                nan_fallback=bool(getattr(args, 'ode_nan_fallback', True)),
                use_unet_checkpoint=bool(getattr(args, 'ode_use_unet_checkpoint', True)),
                disable_solver_fallback=bool(getattr(args, 'ode_disable_solver_fallback', True)),
                local_half=bool(getattr(args, 'ode_local_half', True)),
                freeze_model_params=bool(getattr(args, 'ode_freeze_params', False)),
                empty_cache_each_iter=bool(getattr(args, 'ode_empty_cache_each_iter', True)),
            )
            self.recon_model = DIRE_ODE_Model(args, config, ode_cfg, device=self.device)
            try:
                print("[DIRE_Adv][ode] cfg:", {
                    't_steps': ode_cfg.t_steps,
                    'method': ode_cfg.method,
                    'step_size': ode_cfg.step_size,
                    'rtol': ode_cfg.rtol,
                    'atol': ode_cfg.atol,
                    'fix_rand': ode_cfg.fix_rand,
                    'eot_reps': ode_cfg.eot_reps,
                    'dire_smooth_eps': getattr(ode_cfg, 'dire_smooth_eps', None),
                })
            except Exception:
                pass
        else:
            raise NotImplementedError(f"unknown dire_backend: {self.backend}")

        # Try to put internal diffusion/unet into eval() where possible to reduce nondeterminism
        try:
            # many wrappers keep an inner .model or .ode_runner.model
            if hasattr(self.recon_model, 'model') and self.recon_model.model is not None:
                try:
                    self.recon_model.model.eval()
                except Exception:
                    pass
            if hasattr(self.recon_model, 'ode_runner') and getattr(self.recon_model.ode_runner, 'model', None) is not None:
                try:
                    self.recon_model.ode_runner.model.eval()
                except Exception:
                    pass
            if hasattr(self.recon_model, 'unet') and self.recon_model.unet is not None:
                try:
                    self.recon_model.unet.eval()
                except Exception:
                    pass
        except Exception:
            pass

        # counter & tag
        self.register_buffer('counter', torch.zeros(1, device=self.device))
        self.tag = None

        # cached linearization state
        self._cached_x = None
        self._cached_y = None
        self._cached_y_for_cls = None

        # linearization flag
        self.use_linearized_autattack = bool(getattr(args, 'use_linearized_autattack', False))

        # classifier preprocessing flags (must match export/dataset)
        self.dire_target_size = int(getattr(self.args, 'dire_target_size', 224))
        self.dire_apply_imagenet_norm = bool(getattr(self.args, 'dire_apply_imagenet_norm', True))
        self.dire_scale_to_m11 = bool(getattr(self.args, 'dire_scale_to_m11', False))
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).to(self.device)
        self._imagenet_mean = mean.view(1, 3, 1, 1)
        self._imagenet_std = std.view(1, 3, 1, 1)

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def _prepare_dire_for_classifier(self, dire_01: torch.Tensor) -> torch.Tensor:
        """
        dire_01: [B,C,H,W] in [0,1] (or [C,H,W])
        returns: [B,3,ts,ts] normalized per configuration
        """
        if dire_01.dim() == 3:
            dire_01 = dire_01.unsqueeze(0)
        if dire_01.shape[1] == 1:
            dire_01 = dire_01.repeat(1, 3, 1, 1)

        H, W = dire_01.shape[-2], dire_01.shape[-1]
        crop256 = 256
        if H >= crop256 and W >= crop256:
            dire_256 = TF.center_crop(dire_01, crop256)
        else:
            dire_256 = F.interpolate(dire_01, size=(crop256, crop256), mode='bilinear', align_corners=False)

        ts = int(self.dire_target_size) if self.dire_target_size else 256
        if ts != 256:
            dire_final = TF.center_crop(dire_256, ts)
        else:
            dire_final = dire_256

        if self.dire_apply_imagenet_norm:
            dire_input = (dire_final - self._imagenet_mean) / self._imagenet_std
        elif self.dire_scale_to_m11:
            dire_input = dire_final * 2.0 - 1.0
        else:
            dire_input = dire_final

        return dire_input

    def forward(self, x):
        """
        x: [B, C, H, W] in [0,1] (or [C,H,W])
        returns logits
        """
        counter = int(self.counter.item())
        log_every = getattr(self.args, 'log_prob_every', 10)
        should_log = (counter == 0) or (log_every > 0 and (counter % log_every == 0))
        if should_log:
            print(f'diffusion times: {counter}')

        is_imagenet = ('imagenet' in getattr(self.args, 'domain', ''))
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        if is_imagenet and x.shape[-1] != 256:
            if should_log:
                print(f"[warn] Input size {x.shape[-2:]} != 256, upsampling (this adds interpolation error!)")
            x_in = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            x_in = x

        # linearized bridge path if enabled
        if self.use_linearized_autattack and (self._cached_y is not None) and (self._cached_x is not None):
            logits = _LinearizedLogitsFunction.apply(x_in, self._cached_y_for_cls, self._cached_y, self._cached_x, self.classifier)
        else:
            # full recon
            x_m1_1 = (x_in - 0.5) * 2.0
            start_time = time.time()
            dire = self.recon_model.dire_map(x_m1_1, enable_grad=True, normalize=True)
            dire_01 = torch.clamp(dire, 0.0, 1.0)
            minutes, seconds = divmod(time.time() - start_time, 60)

            dire_input = self._prepare_dire_for_classifier(dire_01)

            if should_log:
                try:
                    print(f'x shape (before recon): {x.shape}')
                    print(f'dire_map shape (before classifier): {dire_01.shape}')
                    _min = float(dire_01.min().item())
                    _max = float(dire_01.max().item())
                    _mean = float(dire_01.mean().item())
                    print(f'dire_map stats: min={_min:.6f} max={_max:.6f} mean={_mean:.6f}')
                    print("DIRE compute time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))
                except Exception:
                    pass

            logits = self.classifier(dire_input)

        # convert logits->2-class if needed (preserve graph)
        def _prob_tensor_to_binary_logits(prob_tensor: torch.Tensor) -> torch.Tensor:
            probs = prob_tensor
            if torch.any((probs < 0) | (probs > 1)):
                probs = torch.sigmoid(probs)
            eps = 1e-6
            probs = probs * (1.0 - 2.0 * eps) + eps
            logits_real = torch.log1p(-probs)
            logits_fake = torch.log(probs)
            return torch.stack([logits_real, logits_fake], dim=1)

        fake_prob_index = getattr(self.args, 'fake_prob_index', None)
        if fake_prob_index is not None and logits.dim() == 2 and logits.shape[1] > 1:
            logits = _prob_tensor_to_binary_logits(logits[:, fake_prob_index])
        elif logits.dim() == 1:
            logits = _prob_tensor_to_binary_logits(logits)
        elif logits.dim() == 2 and logits.shape[1] == 1:
            logits = _prob_tensor_to_binary_logits(logits.view(-1))

        try:
            probs_fake = torch.softmax(logits, dim=1)[:, 1]
            if should_log:
                probs_cpu = probs_fake.detach().cpu().tolist()
                if len(probs_cpu) == 1:
                    print(f'fake prob: {probs_cpu[0]:.6f}')
                else:
                    print(f'fake prob (per-sample): {[round(p, 6) for p in probs_cpu]}  mean={float(torch.tensor(probs_cpu).mean()):.6f}')
                p_low = getattr(self.args, 'prob_low_real', 0.1)
                p_high = getattr(self.args, 'prob_high_fake', 0.9)
                below = (probs_fake < p_low).sum().item()
                above = (probs_fake > p_high).sum().item()
                print(f'soft goals: prob<={p_low} count {below}, prob>={p_high} count {above}')
        except Exception:
            pass

        self.counter += 1
        return logits

    def recon(self, x):
        """
        Return recon image in [0,1]; optional center-crop to 224 via recon_output_224 flag.
        """
        is_imagenet = ('imagenet' in getattr(self.args, 'domain', ''))
        need_224 = bool(getattr(self.args, 'recon_output_224', False))

        x_in = x
        if is_imagenet and x.shape[-1] != 256:
            x_in = F.interpolate(x_in, size=(256, 256), mode='bilinear', align_corners=False)

        x_m1_1 = (x_in - 0.5) * 2.0
        recon_m1_1 = self.recon_model.recon_from_x(x_m1_1, enable_grad=True)
        recon_01 = torch.clamp((recon_m1_1 + 1.0) * 0.5, 0.0, 1.0)

        if is_imagenet and need_224:
            recon_01 = TF.center_crop(recon_01, 224)

        return recon_01

    def linearize_start(self, x):
        """
        Cache DIRE(x) with graph to allow backward via cached graph.
        x: [B,C,H,W] in [0,1] (or [C,H,W])
        """
        # clear
        self._cached_x = None
        self._cached_y = None
        self._cached_y_for_cls = None

        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        is_imagenet = ('imagenet' in getattr(self.args, 'domain', ''))
        if is_imagenet and x.shape[-1] != 256:
            x_in = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            x_in = x

        x_tracked = x_in.detach().clone()
        x_tracked.requires_grad_(True)
        self._cached_x = x_tracked

        x_m1_1 = (x_tracked - 0.5) * 2.0
        y = self.recon_model.dire_map(x_m1_1, enable_grad=True, normalize=True)
        y = torch.clamp(y, 0.0, 1.0)
        self._cached_y = y

        # create cached_y_for_cls via differentiable ops
        tmp = self._cached_y if self._cached_y.dim() == 4 else self._cached_y.unsqueeze(0)
        if tmp.shape[-1] >= 256 and tmp.shape[-2] >= 256:
            tmp256 = TF.center_crop(tmp, 256)
        else:
            tmp256 = F.interpolate(tmp, size=(256, 256), mode='bilinear', align_corners=False)

        ts = int(self.dire_target_size) if self.dire_target_size else 256
        if ts != 256:
            tmp_final = TF.center_crop(tmp256, ts)
        else:
            tmp_final = tmp256

        if tmp_final.shape[1] == 1:
            tmp_final = tmp_final.repeat(1, 3, 1, 1)

        if self.dire_apply_imagenet_norm:
            self._cached_y_for_cls = (tmp_final - self._imagenet_mean) / self._imagenet_std
        elif self.dire_scale_to_m11:
            self._cached_y_for_cls = tmp_final * 2.0 - 1.0
        else:
            self._cached_y_for_cls = tmp_final

        if getattr(self.args, 'log_linearize', False):
            try:
                print("[linearize_start] cached_y stats:", float(self._cached_y.min().item()),
                      float(self._cached_y.max().item()), float(self._cached_y.mean().item()))
            except Exception:
                pass

    def enable_linearized_autattack(self, enable: bool = True):
        self.use_linearized_autattack = enable
