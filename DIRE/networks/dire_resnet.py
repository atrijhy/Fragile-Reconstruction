from __future__ import annotations

import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from DiffPure.runners.dire_ode_model import DIRE_ODE_Model, ODEDireConfig
from DiffPure import utils as diff_utils

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


BASE_ARCHES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


class DireOdeResNet(nn.Module):
    """Wrap a ResNet so it consumes DIRE maps generated via probability-flow ODE."""

    def __init__(
        self,
        base_arch: str,
        cfg,
        is_train: bool = False,
        continue_train: bool = False,
        init_gain: float = 0.02,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if cfg is None:
            raise ValueError("cfg must be provided for DireOdeResNet")
        if base_arch not in BASE_ARCHES:
            raise ValueError(f"Unsupported base arch: {base_arch}")
        # ODE backend is optional when training/evaluating on exported .pt DIRE caches.
        # We only require these when we actually need to compute DIRE online.
        need_ode_backend = bool(getattr(cfg, "dire_mode", False)) and (
            not getattr(cfg, "dire_train_real", "") or not getattr(cfg, "dire_train_fake", "")
        )
        self._ode_available = bool(cfg.dire_config_path) and bool(cfg.dire_diffusion_model_path)
        if need_ode_backend and not self._ode_available:
            raise ValueError(
                "DIRE ODE backend requested but missing cfg.dire_config_path or cfg.dire_diffusion_model_path."
            )

        self.cfg = cfg
        self.cache_enabled = bool(cfg.dire_cache_maps)
        self.cache_store: dict[str, torch.Tensor] = {}

        self.dire_model = None
        if self._ode_available:
            config_path = Path(cfg.dire_config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"dire_config_path not found: {config_path}")
            with open(config_path, "r") as f:
                config_yaml = yaml.safe_load(f)
            config_ns = diff_utils.dict2namespace(config_yaml)

            ode_args = SimpleNamespace(diffusion_model_path=cfg.dire_diffusion_model_path)
            ode_cfg = ODEDireConfig(
                t_steps=int(cfg.dire_t_steps),
                method=str(cfg.dire_ode_method),
                step_size=float(cfg.dire_ode_step_size),
                rtol=float(cfg.dire_ode_rtol),
                atol=float(cfg.dire_ode_atol),
                fix_rand=bool(cfg.dire_fix_rand),
                eot_reps=int(cfg.dire_eot_reps),
                dire_smooth_eps=float(cfg.dire_smooth_eps),
                dire_log_grad=False,
            )
            self.dire_model = DIRE_ODE_Model(ode_args, config_ns, ode_cfg=ode_cfg)

        base_builder = BASE_ARCHES[base_arch]
        if is_train and not continue_train:
            base = base_builder(pretrained=pretrained)
            base.fc = nn.Linear(base.fc.in_features, 1)
            nn.init.normal_(base.fc.weight.data, 0.0, init_gain)
        else:
            base = base_builder(num_classes=1)
        self.base = base

        target_size = int(getattr(cfg, "dire_target_size", 0))
        self.target_size = target_size if target_size > 0 else None
        self.apply_imagenet_norm = bool(getattr(cfg, "dire_apply_imagenet_norm", True))
        self.scale_to_m11 = bool(getattr(cfg, "dire_scale_to_m11", False)) and not self.apply_imagenet_norm  # 确保默认值为 True
        # When True, forward() uses x directly as DIRE input (skip disk/cache/ODE loading).
        # Used for PGD adversarial training on precomputed DIRE maps.
        self.bypass_dire_loading = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)
        # one-time log flag
        self._log_preproc_once = False

    def _prepare_dire_input(self, dire_map: torch.Tensor, already_preprocessed: bool = False) -> torch.Tensor:
        if dire_map.dim() == 3:
            dire_map = dire_map.unsqueeze(0)

        if already_preprocessed:
            return dire_map

        if dire_map.shape[1] == 1:
            dire_map = dire_map.repeat(1, 3, 1, 1)
        elif dire_map.shape[1] != 3:
            raise ValueError(f"Unexpected DIRE map channels: {dire_map.shape[1]}")

        if self.target_size:
            H, W = dire_map.shape[-2], dire_map.shape[-1]
            tgt = int(self.target_size)
            if (H, W) == (tgt, tgt):
                pass
            elif H >= tgt and W >= tgt:
                # Prefer CenterCrop when reducing from 256->224 to match eval/attack pipeline
                import torchvision.transforms.functional as TF
                dire_map = TF.center_crop(dire_map, tgt)
            else:
                # If somehow smaller, upscale with bilinear
                dire_map = F.interpolate(dire_map, size=(tgt, tgt), mode="bilinear", align_corners=False)

        dire_map = dire_map.clamp(0.0, 1.0)
        if self.scale_to_m11:
            dire_map = dire_map * 2.0 - 1.0
        elif self.apply_imagenet_norm:
            dire_map = (dire_map - self.imagenet_mean) / self.imagenet_std
        return dire_map

    def maybe_cache(self, key: Optional[str], value: torch.Tensor) -> None:
        if not self.cache_enabled or key is None:
            return
        self.cache_store[key] = value.detach().cpu()

    def pull_cache(self, key: Optional[str], device: torch.device) -> Optional[torch.Tensor]:
        if not self.cache_enabled or key is None:
            return None
        cached = self.cache_store.get(key)
        if cached is None:
            return None
        return cached.to(device)

    def forward(self, x: torch.Tensor, meta: Optional[dict] = None) -> torch.Tensor:
        device = x.device
        if not self._log_preproc_once:
            mode = 'imagenet_norm' if self.apply_imagenet_norm else ('scale_to_m11' if self.scale_to_m11 else 'raw_01')
            bypass = ' | bypass_dire_loading=True' if self.bypass_dire_loading else ''
            try:
                print(f"[DireOdeResNet] preproc mode={mode}, target_size={self.target_size or 'none'} | ODE available={self.dire_model is not None}{bypass}")
            except Exception:
                pass
            self._log_preproc_once = True

        # PGD mode: x is already a preprocessed DIRE map (perturbed + scale_to_m11 etc.)
        # Skip all cache/meta/ODE loading and feed x directly to classifier.
        if self.bypass_dire_loading:
            return self.base(x)

        cache_key = None
        cache_tensor = None
        cache_path: Optional[str] = None
        preprocessed = False
        if meta:
            cache_key = meta.get("path")
            cache_tensor = meta.get("dire_map_tensor")
            cache_path = meta.get("dire_pt_path")
            # 'dire_preprocessed' may be a bool, a tensor (batched), or a list.
            val = meta.get("dire_preprocessed", False)
            if isinstance(val, torch.Tensor):
                # if batched tensor, take first element; if scalar tensor, take its item
                try:
                    preprocessed = bool(val.item()) if val.numel() == 1 else bool(val[0].item())
                except Exception:
                    preprocessed = False
            elif isinstance(val, (list, tuple)):
                preprocessed = bool(val[0]) if len(val) > 0 else False
            else:
                preprocessed = bool(val)
            if isinstance(cache_key, list):
                # batching can provide list of paths; caching expects str keys
                cache_key = None

        cached = self.pull_cache(cache_key, device)
        if cached is not None:
            dire_input = cached
        elif cache_tensor is not None:
            dire_map = cache_tensor.to(device)
            dire_input = self._prepare_dire_input(dire_map, already_preprocessed=preprocessed)
            self.maybe_cache(cache_key, dire_input)
        elif cache_path is not None:
            loaded = torch.load(cache_path, map_location="cpu")
            dire_map = loaded["dire_map"].float().to(device)
            dire_input = self._prepare_dire_input(dire_map, already_preprocessed=False)
            self.maybe_cache(cache_key, dire_input)
        else:
            if self.dire_model is None:
                raise RuntimeError(
                    "DIRE ODE backend not initialized. Provide dire_config_path and dire_diffusion_model_path, "
                    "or supply precomputed dire_map via meta (dire_map_tensor or dire_pt_path)."
                )
            # Ensure ODE computed on 256×256 like eval/attack. Dataset may have produced 224×224.
            x_in = x
            if x_in.shape[-1] != 256:
                # Log once per process
                try:
                    print(f"[DireOdeResNet][warn] Input size {tuple(x_in.shape[-2:])} != 256; upsampling to 256 for ODE (to match eval pipeline)")
                except Exception:
                    pass
                x_in = F.interpolate(x_in, size=(256, 256), mode="bilinear", align_corners=False)
            # Sanity: ODE expects [-1,1] domain. If not, warn.
            try:
                xmin = float(x_in.min().item())
                xmax = float(x_in.max().item())
                if xmin < -1.5 or xmax > 1.5:
                    print(f"[DireOdeResNet][warn] input range to ODE seems off: [{xmin:.3f}, {xmax:.3f}] (expected ~[-1,1])")
            except Exception:
                pass
            dire_out = self.dire_model.forward(
                x_in,
                t_steps=int(self.cfg.dire_t_steps),
                fix_rand=bool(self.cfg.dire_fix_rand),
                eot_reps=int(self.cfg.dire_eot_reps),
            )
            dire_map = dire_out["dire_map"]
            dire_input = self._prepare_dire_input(dire_map, already_preprocessed=False)
            self.maybe_cache(cache_key, dire_input)
        logits = self.base(dire_input)
        return logits
