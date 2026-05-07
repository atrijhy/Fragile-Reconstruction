# dire_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAP_EXTS = {".pt", ".pth"}
VALID_EXTS = IMG_EXTS | MAP_EXTS


def _collect_images(root: str | Path) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    files: List[Path] = []
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files)


class DireRealFakeDataset(Dataset):
    """
    Dataset that returns processed RGB tensors along with binary labels.

    Design choices:
      - Do NOT shuffle in constructor. Keep deterministic sorted order.
      - Support both image files and .pt/.pth dire cache files.
      - Normalize with ImageNet mean/std on the same device as the tensor at usage time.
    """

    def __init__(
        self,
        real_dirs: List[str],
        fake_dirs: List[str],
        resize: int = 256,
        crop: int = 256,
        seed: int = 42,  # kept for API compatibility but not used for in-constructor shuffle
        target_size: Optional[int] = 224,
        base_256: bool = True,
        scale_to_m11: bool = False,
        apply_imagenet_norm: bool = True,
        # augmentation controls (effective during training when enabled)
        is_train: bool = False,
        aug_random_crop224: bool = False,
        aug_flip: bool = False,
        flip_prob: float = 0.5,
        interp: str = "bilinear",
        paths: Optional[List[Path]] = None,
        labels: Optional[List[int]] = None,
        # DIRE map normalization: scale to [0, 1] range for proper adversarial training
        normalize_dire: bool = False,
        dire_global_max: float = 0.2,  # approximate max value of DIRE maps
        # Raw input mode: keep tensor in [0, 1] range, no preprocessing
        # For input-space PGD where preprocessing is done after perturbation
        raw_input: bool = False,
        # 限制每类样本数量（0 表示不限制）
        max_samples: int = 0,
    ) -> None:
        super().__init__()

        if paths is None or labels is None:
            self.real_paths: List[Path] = []
            self.fake_paths: List[Path] = []
            for d in real_dirs:
                self.real_paths.extend(_collect_images(d))
            for d in fake_dirs:
                self.fake_paths.extend(_collect_images(d))

            if not self.real_paths:
                raise FileNotFoundError(f"No real images found in: {real_dirs}")
            if not self.fake_paths:
                raise FileNotFoundError(f"No fake images found in: {fake_dirs}")

            # 限制每类样本数量
            if max_samples > 0:
                self.real_paths = self.real_paths[:max_samples]
                self.fake_paths = self.fake_paths[:max_samples]

            all_paths = self.real_paths + self.fake_paths
            all_labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)

            # Keep deterministic sorted order; DO NOT shuffle here.
            self.paths = list(all_paths)
            self.labels = list(all_labels)
        else:
            assert len(paths) == len(labels), "paths and labels must have the same length"
            self.paths = list(paths)
            self.labels = list(labels)

        self.target_size = int(target_size) if target_size and int(target_size) > 0 else None
        self.base_256 = bool(base_256)
        self.raw_input = bool(raw_input)
        # When raw_input=True, keep tensor in [0, 1] range (no normalization)
        # This is for input-space PGD where preprocessing is done after perturbation
        if self.raw_input:
            self.apply_imagenet_norm = False
            self.scale_to_m11 = False
        else:
            self.apply_imagenet_norm = bool(apply_imagenet_norm)
            self.scale_to_m11 = bool(scale_to_m11) and not self.apply_imagenet_norm
        self.is_train = bool(is_train)
        self.aug_random_crop224 = bool(aug_random_crop224)
        self.aug_flip = bool(aug_flip)
        self.flip_prob = float(flip_prob)
        self.interp = (interp or "bilinear").lower()
        # DIRE map normalization for adversarial training
        self.normalize_dire = bool(normalize_dire)
        self.dire_global_max = float(dire_global_max)
        _tv_map = {
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "lanczos": transforms.InterpolationMode.LANCZOS,
            "nearest": transforms.InterpolationMode.NEAREST,
        }
        self._tv_interp = _tv_map.get(self.interp, transforms.InterpolationMode.BILINEAR)
        # TF.resize on Tensors does NOT support LANCZOS; fall back to BICUBIC for tensor path
        self._tensor_interp = (
            transforms.InterpolationMode.BICUBIC
            if self._tv_interp == transforms.InterpolationMode.LANCZOS
            else self._tv_interp
        )

        # store mean/std as lists and create tensors on-use to avoid device mismatch
        self._imagenet_mean = [0.485, 0.456, 0.406]
        self._imagenet_std = [0.229, 0.224, 0.225]

        # transforms used for image-file branch
        # Always: Resize(256) (+ optional CenterCrop(256) to get square) + ToTensor
        # Random crop/flip are applied in tensor space inside _finalize_tensor for BOTH PNG and .pt inputs.
        transform_list = [transforms.Resize(resize, interpolation=self._tv_interp)]
        if crop:
            transform_list.append(transforms.CenterCrop(crop))
        transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)

        print(
            f"[DireRealFakeDataset] num_samples={len(self.paths)}, target_size={self.target_size}, "
            f"base_256={self.base_256}, raw_input={self.raw_input}, "
            f"apply_imagenet_norm={self.apply_imagenet_norm}, scale_to_m11={self.scale_to_m11}, "
            f"is_train={self.is_train}, aug_random_crop224={self.aug_random_crop224}, "
            f"aug_flip={self.aug_flip}, flip_prob={self.flip_prob}, interp={self.interp}, "
            f"normalize_dire={self.normalize_dire}, dire_global_max={self.dire_global_max}"
        )

    def _resize_tensor(self, tensor: torch.Tensor, size: List[int] | Tuple[int, int]) -> torch.Tensor:
        """Resize a CHW tensor using a tensor-safe interpolation (no LANCZOS)."""
        return TF.resize(tensor, size=size, interpolation=self._tensor_interp, antialias=True)

    def _center_crop_256_then_target(self, tensor: torch.Tensor) -> torch.Tensor:
        C, H, W = tensor.shape
        crop256 = 256
        if H >= crop256 and W >= crop256:
            top = (H - crop256) // 2
            left = (W - crop256) // 2
            tensor = tensor[:, top:top + crop256, left:left + crop256]
        else:
            # Resize tensors with a safe mode (no LANCZOS in tensor path)
            tensor = self._resize_tensor(tensor, size=[crop256, crop256])
        if self.target_size and int(self.target_size) != crop256:
            ts = int(self.target_size)
            top = (crop256 - ts) // 2
            left = (crop256 - ts) // 2
            tensor = tensor[:, top:top + ts, left:left + ts]
        return tensor

    def _ensure_256_square(self, tensor: torch.Tensor) -> torch.Tensor:
        C, H, W = tensor.shape
        crop256 = 256
        if H >= crop256 and W >= crop256:
            # Center-crop to 256x256
            top = (H - crop256) // 2
            left = (W - crop256) // 2
            tensor = tensor[:, top:top + crop256, left:left + crop256]
        else:
            # Resize up to 256x256 using configured interpolation (tensor-safe)
            tensor = self._resize_tensor(tensor, size=[crop256, crop256])
        return tensor

    def _finalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # DIRE map normalization: scale from [0, dire_global_max] to [0, 1]
        # This should be done BEFORE clamping so that eps in [0,1] space is meaningful
        if self.normalize_dire:
            tensor = tensor.float() / self.dire_global_max
            tensor = tensor.clamp(0.0, 1.0)  # clamp after scaling
        else:
            tensor = tensor.float().clamp(0.0, 1.0)
        ts = int(self.target_size) if self.target_size else 224
        if self.base_256:
            # Path A: unify to 256x256 baseline then crop target
            tensor = self._ensure_256_square(tensor)
            if self.is_train and self.aug_random_crop224 and ts > 0:
                max_off = 256 - ts
                if max_off < 0:
                    tensor = self._resize_tensor(tensor, size=[ts, ts])
                else:
                    top = random.randint(0, max_off)
                    left = random.randint(0, max_off)
                    tensor = tensor[:, top:top + ts, left:left + ts]
            else:
                if ts != 256:
                    top = (256 - ts) // 2
                    left = (256 - ts) // 2
                    tensor = tensor[:, top:top + ts, left:left + ts]
        else:
            # Path B: keep native resolution (e.g., 64x64) and crop/resize directly to target_size
            C, H, W = tensor.shape
            if ts and ts > 0:
                if self.is_train and self.aug_random_crop224:
                    if H >= ts and W >= ts:
                        max_top = max(0, H - ts)
                        max_left = max(0, W - ts)
                        top = random.randint(0, max_top)
                        left = random.randint(0, max_left)
                        tensor = tensor[:, top:top + ts, left:left + ts]
                    else:
                        tensor = self._resize_tensor(tensor, size=[ts, ts])
                else:
                    if H == ts and W == ts:
                        pass
                    elif H >= ts and W >= ts:
                        top = (H - ts) // 2
                        left = (W - ts) // 2
                        tensor = tensor[:, top:top + ts, left:left + ts]
                    else:
                        tensor = self._resize_tensor(tensor, size=[ts, ts])
        # Optional horizontal flip (tensor space) during training
        if self.is_train and self.aug_flip and random.random() < float(self.flip_prob):
            tensor = torch.flip(tensor, dims=[2])  # horizontal flip (W dimension)
        if self.apply_imagenet_norm:
            mean = torch.tensor(self._imagenet_mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
            std = torch.tensor(self._imagenet_std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
            tensor = (tensor - mean) / std
        elif self.scale_to_m11:
            tensor = tensor * 2.0 - 1.0
        return tensor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        path = Path(self.paths[idx])
        label = int(self.labels[idx])
        suffix = path.suffix.lower()

        if suffix in MAP_EXTS:
            data = torch.load(path, map_location="cpu")
            if "dire_map" not in data:
                raise KeyError(f"dire_map not found in cache file: {path}")
            dire_map = data["dire_map"]
            if not isinstance(dire_map, torch.Tensor):
                dire_map = torch.tensor(dire_map)

            if dire_map.ndim == 4 and dire_map.shape[0] == 1:
                dire_map = dire_map.squeeze(0)
            if dire_map.ndim == 2:
                dire_map = dire_map.unsqueeze(0)
            if dire_map.ndim != 3:
                raise ValueError(f"Unexpected dire_map shape {tuple(dire_map.shape)} for file {path}")

            C = dire_map.shape[0]
            if C == 1:
                tensor = dire_map.repeat(3, 1, 1).float()
            elif C == 3:
                tensor = dire_map.float()
            else:
                raise ValueError(f"Unsupported dire_map channels {C} for file {path}")

            tensor = tensor.clamp(0.0, 1.0)
            tensor = self._finalize_tensor(tensor)
            meta = {"path": str(data.get("source", str(path))), "dire_pt_path": str(path)}
        else:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            tensor = self._finalize_tensor(tensor)
            meta = {"path": str(path)}

        return tensor, torch.tensor(label, dtype=torch.long), meta
