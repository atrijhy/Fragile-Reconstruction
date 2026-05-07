#!/usr/bin/env python3
"""
Unified detector module for DIRE, LaRE, and AeroBlade.

All evaluation scripts import detectors from here to ensure implementation consistency.

Usage:
    from shared_detectors import DIREDetector, LaREDetector, AEDetector

PYTHONPATH setup (set before running any script that imports this):
    export PYTHONPATH=/path/to/release_code/DIRE:/path/to/release_code/LaRE:/path/to/release_code/aeroblade/src:/path/to/release_code/DiffPure:$PYTHONPATH

Required checkpoints (set via env vars or pass directly to load_models()):
    DIRE_CKPT_ROOT  - directory containing cls_{gen}_t20/ckpt/model_epoch_best.pth
    DIFFUSION_CKPT  - path to 256x256_diffusion_uncond.pt
    LARE_CKPT_ROOT  - directory containing ExpLaRE2_{gen}_*/Val_best.pth
    SD15_MODEL_PATH - path to SD 1.5 HuggingFace model folder
    AE_CALIB_CSV    - path to aeroblade calibration distances CSV
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

AE_GENS = ["adm", "flux", "sdv5", "vqdm"]


# ==============================================================================
# DIRE Detector
# ==============================================================================
class DIREDetector:
    """
    DIRE Detector: image -> ODE -> DIRE map -> classifier -> prediction

    Standard config:
    - t=20 (diffusion timesteps)
    - dire_image_size=256
    - classifier: dire_resnet50_m11
    """

    def __init__(self, device: str = "cuda", t: int = 20):
        self.device = torch.device(device)
        self.t = t
        self.runner = None
        self.classifier = None
        self.current_gen = None

    def load_models(self, gen: str, ckpt_root: Optional[str] = None, diffusion_ckpt: Optional[str] = None, config_path: Optional[str] = None):
        """Load DIRE ODE model and classifier for specific generator.

        Args:
            gen: generator name (adm, flux, sdv5, vqdm)
            ckpt_root: root directory for classifier checkpoints
                       (expected: <ckpt_root>/cls_{gen}_t20/ckpt/model_epoch_best.pth)
                       Defaults to env var DIRE_CKPT_ROOT.
            diffusion_ckpt: path to 256x256_diffusion_uncond.pt
                            Defaults to env var DIFFUSION_CKPT.
            config_path: path to imagenet.yml
                         Defaults to ./DiffPure/configs/imagenet.yml relative to this file.
        """
        if self.current_gen == gen and self.runner is not None:
            return

        print(f"\n[DIRE] Loading models for {gen}...")
        import yaml
        from utils import dict2namespace, get_image_classifier
        from runners.dire_ode_model import DIRE_ODE_Model, ODEDireConfig

        if config_path is None:
            config_path = os.environ.get(
                "DIFFPURE_CONFIG",
                str(Path(__file__).parent / "DiffPure" / "configs" / "imagenet.yml"),
            )
        with open(config_path, "r") as f:
            config = dict2namespace(yaml.safe_load(f))
        config.device = self.device

        class Args:
            pass

        args = Args()
        args.t = self.t
        args.seed = 42
        args.fix_rand = True
        args.eot_reps = 1
        args.ode_method = "euler"
        args.ode_rtol = 1e-3
        args.ode_atol = 1e-3
        args.ode_step_size = 2e-2
        args.ode_dire_smooth_eps = 5e-5
        args.dire_image_size = 256
        args.config = str(config_path)

        if diffusion_ckpt is None:
            diffusion_ckpt = os.environ.get("DIFFUSION_CKPT", "/path/to/256x256_diffusion_uncond.pt")
        args.diffusion_model_path = diffusion_ckpt

        ode_cfg = ODEDireConfig(
            t_steps=self.t,
            method="euler",
            rtol=1e-3,
            atol=1e-3,
            step_size=2e-2,
            fix_rand=True,
            eot_reps=1,
            clamp_recon=True,
            dire_scale_to_01=True,
            dire_smooth_eps=5e-5,
            dire_log_grad=False,
            force_fp32=True,
            nan_fallback=True,
            use_unet_checkpoint=True,
            disable_solver_fallback=True,
            local_half=False,
            freeze_model_params=False,
            empty_cache_each_iter=True,
        )

        if self.runner is None:
            self.runner = DIRE_ODE_Model(args, config, ode_cfg=ode_cfg, device=self.device)
            self.runner.eval()

        if ckpt_root is None:
            ckpt_root = os.environ.get("DIRE_CKPT_ROOT", "/path/to/dire/checkpoints")
        ckpt_path = Path(ckpt_root) / f"cls_{gen}_t20" / "ckpt" / "model_epoch_best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(ckpt_root) / f"cls_{gen}_t{self.t}" / "ckpt" / "model_epoch_best.pth"

        self.classifier = get_image_classifier("dire_resnet50_m11", ckpt_path=str(ckpt_path))
        self.classifier.to(self.device).eval()
        self.current_gen = gen
        print(f"[DIRE] Loaded classifier: {ckpt_path}")

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        map_scale: float = 1.0,
        classifier_feature_scale: float = 1.0,
        feature_scale: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict real/fake for images.

        Args:
            images: [B, 3, H, W] tensor in [0, 1]
            map_scale: scale applied to DIRE map before classifier (default 1.0).
            classifier_feature_scale: scale applied to the classifier's last-layer feature.
            feature_scale: if set, overrides both map_scale and classifier_feature_scale.

        Returns:
            predictions: [B] tensor of 0 (real) or 1 (fake)
            probabilities: [B] tensor of fake probability
        """
        if feature_scale is not None:
            map_scale = feature_scale
            classifier_feature_scale = 1.0
        images = images.to(self.device)

        x = (images - 0.5) * 2.0
        out = self.runner.forward(x, t_steps=self.t, eot_reps=1, fix_rand=True)
        dire_map = out["dire_map"]

        dire_map = TF.resize(dire_map, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
        dire_map = TF.center_crop(dire_map, 224)
        if map_scale != 1.0:
            dire_map = dire_map * map_scale

        if classifier_feature_scale != 1.0:
            scale = float(classifier_feature_scale)
            def _pre_hook(module, inp):
                return (inp[0] * scale,)
            handle = self.classifier.m.fc.register_forward_pre_hook(_pre_hook)
            try:
                logits = self.classifier(dire_map).squeeze(-1)
            finally:
                handle.remove()
        else:
            logits = self.classifier(dire_map).squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        if preds.dim() == 0:
            preds = preds.unsqueeze(0)
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        return preds, probs

    @torch.no_grad()
    def get_last_layer_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the classifier's last-layer feature (input to fc), shape [B, feat_dim]."""
        images = images.to(self.device)
        x = (images - 0.5) * 2.0
        out = self.runner.forward(x, t_steps=self.t, eot_reps=1, fix_rand=True)
        dire_map = out["dire_map"]
        dire_map = TF.resize(dire_map, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
        dire_map = TF.center_crop(dire_map, 224)
        captured: List[torch.Tensor] = []

        def _hook(module, inp):
            captured.append(inp[0].detach())

        handle = self.classifier.m.fc.register_forward_pre_hook(_hook)
        try:
            self.classifier(dire_map)
        finally:
            handle.remove()
        return captured[0]  # [B, 2048]

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, batch_size: int = 4, return_scores: bool = False):
        """Evaluate on dataset."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        y_true, y_pred, y_scores = [], [], []
        for images, labels, _ in tqdm(loader, desc="DIRE eval", leave=False):
            images = images.to(self.device)
            preds, probs = self.predict(images)

            y_pred.extend(np.atleast_1d(preds.cpu().numpy()).tolist())
            y_true.extend(np.atleast_1d(labels.numpy()).tolist())
            y_scores.extend(np.atleast_1d(probs.cpu().numpy()).tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        acc = (y_pred == y_true).mean() * 100

        if return_scores:
            return acc, y_true, y_scores
        return acc


# ==============================================================================
# LaRE Detector
# ==============================================================================
class LaREDetector:
    """
    LaRE Detector: image -> SD loss map -> CLIP classifier -> prediction

    Standard config:
    - t=200 (timestep for noise)
    - ensemble_size=4
    - sd_img_size=256 (SD input)
    - data_size=224 (CLIP input)
    - prompt="a photo"
    """

    def __init__(self, device: str = "cuda", t: int = 200, ensemble_size: int = 4):
        self.device = torch.device(device)
        self.t = t
        self.ensemble_size = ensemble_size
        self.sd_img_size = 256
        self.data_size = 224
        self.sd_loaded = False
        self.classifier = None
        self.current_gen = None
        self.sd_path = os.environ.get("SD15_MODEL_PATH", "/path/to/sd15")

    def load_sd_components(self):
        return self.load_sd()

    def load_sd(self):
        """Load SD components for loss map computation."""
        if self.sd_loaded:
            return

        print("[LaRE] Loading SD components...")
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.vae = AutoencoderKL.from_pretrained(self.sd_path, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(self.sd_path, subfolder="unet").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(self.sd_path, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.sd_path, subfolder="tokenizer")
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.sd_path, subfolder="scheduler")

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.sd_loaded = True

    def load_classifier(self, gen: str, ckpt_root: Optional[str] = None):
        """Load LaRE CLIP classifier.

        Args:
            gen: generator name
            ckpt_root: root directory for LaRE checkpoints.
                       Defaults to env var LARE_CKPT_ROOT.
        """
        if self.current_gen == gen and self.classifier is not None:
            return

        print(f"[LaRE] Loading classifier for {gen}...")
        from cross_autoattack_lare_e2e import load_classifier

        if ckpt_root is None:
            ckpt_root = os.environ.get("LARE_CKPT_ROOT", "/path/to/lare/checkpoints")

        ckpt_path = Path(ckpt_root) / f"ExpLaRE2_{gen}_retrain_Log_v12052017" / "Val_best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(ckpt_root) / f"ExpLaRE2_{gen}_retrain_Log_v12060405" / "Val_best.pth"

        self.classifier = load_classifier("CLipClassifierWMapV6", "RN50", str(ckpt_path), self.device)
        self.current_gen = gen
        print(f"[LaRE] Loaded: {ckpt_path}")

    @torch.no_grad()
    def compute_loss_map(self, images: torch.Tensor, clsnames: List[str] = None) -> torch.Tensor:
        """Compute LaRE loss map."""
        bsz = images.shape[0]
        imgs_sd = (images * 2.0 - 1.0).to(self.device)
        if self.sd_img_size > 0:
            imgs_sd = F.interpolate(imgs_sd, size=(self.sd_img_size, self.sd_img_size), mode="bilinear", align_corners=False)

        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)

            latents_dist = self.vae.encode(imgs_sd).latent_dist
            eps = torch.randn_like(latents_dist.mean)
            latents = latents_dist.mean + latents_dist.std * eps
            latents = latents * self.vae.config.scaling_factor

            latents = latents.repeat_interleave(self.ensemble_size, dim=0)
            noise = torch.randn_like(latents)
            timesteps = torch.full((latents.shape[0],), self.t, device=self.device, dtype=torch.long)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        prompts = ["a photo"] * (bsz * self.ensemble_size)
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        encoder_hidden_states = self.text_encoder(text_inputs["input_ids"])[0]

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.reshape(bsz, self.ensemble_size, *loss.shape[1:]).mean(dim=1)

        return loss

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        map_scale: float = 1.0,
        classifier_feature_scale: float = 1.0,
        feature_scale: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict real/fake.

        Args:
            images: [B, 3, H, W] in [0, 1]
            map_scale: scale applied to loss map.
            classifier_feature_scale: scale applied to last-layer feature.
            feature_scale: if set, overrides map_scale, sets classifier_feature_scale=1.

        Returns:
            predictions: [B] tensor of 0 (real) or 1 (fake)
            probabilities: [B] fake probability
        """
        if feature_scale is not None:
            map_scale = feature_scale
            classifier_feature_scale = 1.0
        images = images.to(self.device)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        images_norm = (images - mean) / std
        loss_map = self.compute_loss_map(images)
        if map_scale != 1.0:
            loss_map = loss_map * map_scale

        if classifier_feature_scale != 1.0:
            scale = float(classifier_feature_scale)
            def _pre_hook(module, inp):
                return (inp[0] * scale,)
            handle = self.classifier.fc.register_forward_pre_hook(_pre_hook)
            try:
                logits = self.classifier(images_norm, loss_map)
            finally:
                handle.remove()
        else:
            logits = self.classifier(images_norm, loss_map)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        return preds, probs

    @torch.no_grad()
    def get_last_layer_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the classifier's last-layer feature (input to fc), shape [B, feat_dim]."""
        images = images.to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images_norm = (images - mean) / std
        loss_map = self.compute_loss_map(images)
        captured: List[torch.Tensor] = []

        def _hook(module, inp):
            captured.append(inp[0].detach())

        handle = self.classifier.fc.register_forward_pre_hook(_hook)
        try:
            self.classifier(images_norm, loss_map)
        finally:
            handle.remove()
        return captured[0]  # [B, 3072] for CLipClassifierWMapV6

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, batch_size: int = 4, return_scores: bool = False):
        """Evaluate on dataset."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        y_true, y_pred, y_scores = [], [], []
        for images, labels, clsnames in tqdm(loader, desc="LaRE eval", leave=False):
            preds, probs = self.predict(images)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())
            y_scores.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        acc = (y_pred == y_true).mean() * 100

        if return_scores:
            return acc, y_true, y_scores
        return acc


# ==============================================================================
# AeroBlade (AE) Detector
# ==============================================================================
import ast
import re


class _AEImageDataset(Dataset):
    """Minimal dataset for loading images from a directory."""

    def __init__(self, dir_path: Path, data_size: int = 512, limit: Optional[int] = None):
        self.data_size = data_size
        self.img_paths: List[Path] = []
        for ext in ["**/*.png", "**/*.PNG", "**/*.jpg", "**/*.jpeg"]:
            self.img_paths.extend(dir_path.glob(ext))
        self.img_paths = sorted(set(self.img_paths))
        if limit and limit > 0:
            self.img_paths = self.img_paths[:limit]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.data_size, self.data_size))
        img = img.resize((self.data_size, self.data_size), Image.BILINEAR)
        return TF.to_tensor(img)


class AEDetector:
    """
    AeroBlade Detector: image -> VAE reconstruction -> LPIPS distance -> threshold

    Follows official aeroblade pipeline:
    - SD1.5 VAE, float16 on GPU
    - retrieve_latents + seeded torch.Generator
    - PNG uint8 quantization: mul(255).byte() truncation
    - _PatchedLPIPS(spatial=True) + retPerLayer=True, normalize=True
    - "lpips_vgg_2" = layers_batch[1], negated, spatial average
    """

    def __init__(self, device: str = "cuda", seed: int = 42):
        self.device = torch.device(device)
        self.vae = None
        self.lpips_model = None
        self.data_size = 512
        self.threshold = None
        self.threshold_sense = "ge"
        self.seed = seed
        self.sd_path = os.environ.get("SD15_MODEL_PATH", "/path/to/sd15")
        self._generator = None

    def load_models(self, sd_path: Optional[str] = None):
        """Load VAE and LPIPS models.

        Args:
            sd_path: path to SD 1.5 model folder. Defaults to env var SD15_MODEL_PATH.
        """
        if self.vae is not None:
            return

        if sd_path is not None:
            self.sd_path = sd_path

        print("[AE] Loading models...")
        from diffusers import AutoencoderKL
        from aeroblade.distances import _PatchedLPIPS

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.vae = AutoencoderKL.from_pretrained(
            self.sd_path, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.eval()

        self.lpips_model = _PatchedLPIPS(net="vgg", spatial=True).to(self.device)
        self.lpips_model.eval()

        self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        print("[AE] Models loaded")

    def reset_generator(self):
        """Reset the VAE sampling generator."""
        self._generator = torch.Generator(device=self.device).manual_seed(self.seed)

    @torch.no_grad()
    def compute_distances(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute per-batch reconstruction distances (official aeroblade pipeline).

        Returns:
            [B] tensor — negative LPIPS layer-2.
        """
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )

        images = images.to(self.device)
        if images.shape[-1] != self.data_size or images.shape[-2] != self.data_size:
            images = TF.resize(
                images, [self.data_size, self.data_size],
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            )

        encode_dtype = next(iter(self.vae.parameters())).dtype
        x = images.to(dtype=encode_dtype) * 2.0 - 1.0

        latents = retrieve_latents(self.vae.encode(x), generator=self._generator)
        recon = self.vae.decode(latents, return_dict=False)[0]
        recon = (recon / 2 + 0.5).clamp(0, 1)

        recon = recon.mul(255).byte().float() / 255.0

        images_f32 = images.float()
        recon_f32 = recon.float()

        _sum, layers_batch = self.lpips_model(
            images_f32, recon_f32,
            retPerLayer=True,
            normalize=True,
        )

        dist = -layers_batch[1]
        dist = dist.mean(dim=(2, 3)).view(-1)
        return dist

    @torch.no_grad()
    def compute_dir_distances(
        self, dir_path: Path, batch_size: int = 4, limit: Optional[int] = None,
    ) -> np.ndarray:
        """Compute distances for all images in a directory."""
        self.reset_generator()
        ds = _AEImageDataset(dir_path, data_size=self.data_size, limit=limit)
        if len(ds) == 0:
            return np.array([], dtype=np.float64)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        dists_list: List[float] = []
        for images in tqdm(loader, desc=f"AE {dir_path.name}", leave=False):
            d = self.compute_distances(images)
            dists_list.extend(d.cpu().numpy().tolist())
        return np.array(dists_list, dtype=np.float64)

    def load_threshold_from_csv(self, gen: str, csv_path: Optional[str] = None):
        """Load threshold from precomputed calibration CSV."""
        if csv_path is None:
            csv_path = os.environ.get("AE_CALIB_CSV", "/path/to/distances_calib.csv")
        csv_p = Path(csv_path)
        if csv_p.exists():
            try:
                return self._load_threshold_from_csv(csv_p, gen)
            except Exception as e:
                print(f"[AE] CSV load failed: {e}")
                return None
        return None

    def find_threshold(self, dataset: Dataset, train_gen: str, batch_size: int = 4, csv_path: Optional[str] = None) -> Tuple[float, str]:
        """Find optimal threshold. Loads from CSV if available, else computes online."""
        if csv_path is None:
            csv_path = os.environ.get("AE_CALIB_CSV", "")

        if csv_path:
            csv_p = Path(csv_path)
            if csv_p.exists():
                try:
                    return self._load_threshold_from_csv(csv_p, train_gen)
                except Exception as e:
                    print(f"[AE] CSV load failed: {e}, computing online...")

        return self._compute_threshold_online(dataset, batch_size)

    def _load_threshold_from_csv(self, csv_path: Path, train_gen: str) -> Tuple[float, str]:
        """Load threshold from CSV (distances stored as negative LPIPS)."""
        import csv as _csv

        print(f"[AE] Loading distances from CSV: {csv_path}")
        dists_list, labels_list = [], []

        with csv_path.open("r", encoding="utf-8") as fh:
            reader = _csv.reader(fh)
            header = next(reader, None)
            if header:
                col_map = {h.strip(): i for i, h in enumerate(header)}
                dir_idx = col_map.get("dir", 0)
                repo_idx = col_map.get("repo_id", None)
                dist_idx = col_map.get("distance", 6)
            else:
                dir_idx, repo_idx, dist_idx = 0, None, 6

            for row in reader:
                if len(row) <= max(dir_idx, dist_idx):
                    continue
                dir_col = row[dir_idx]
                try:
                    dist_val = float(row[dist_idx])
                except Exception:
                    continue

                if repo_idx is not None and repo_idx < len(row):
                    rid = row[repo_idx].strip()
                    if rid != "max":
                        continue

                if f"/{train_gen}/" in dir_col or dir_col.endswith(f"/{train_gen}/test") or dir_col.endswith(f"/{train_gen}/val"):
                    labels_list.append(1)
                    dists_list.append(dist_val)
                elif "/real/" in dir_col or dir_col.endswith("/real/test") or dir_col.endswith("/real/val"):
                    labels_list.append(0)
                    dists_list.append(dist_val)

        if len(dists_list) == 0:
            raise ValueError(f"No distances for {train_gen} in CSV")

        dists = np.array(dists_list)
        labels = np.array(labels_list)

        threshold, sense = self._find_optimal_threshold(dists, labels)
        self.threshold = threshold
        self.threshold_sense = sense
        calib_preds = (dists >= threshold).astype(int) if sense == "ge" else (dists <= threshold).astype(int)
        calib_acc = (calib_preds == labels).mean() * 100
        print(f"[AE] CSV Threshold: {threshold:.6f}, sense={sense}, acc={calib_acc:.2f}%")
        return self.threshold, self.threshold_sense

    def _compute_threshold_online(self, dataset: Dataset, batch_size: int) -> Tuple[float, str]:
        """Compute threshold online from dataset."""
        self.reset_generator()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_dists, all_labels = [], []
        for images, labels, _ in tqdm(loader, desc="AE threshold", leave=False):
            dists = self.compute_distances(images)
            all_dists.extend(dists.cpu().numpy())
            all_labels.extend(labels.numpy())

        dists = np.array(all_dists)
        labels = np.array(all_labels)

        threshold, sense = self._find_optimal_threshold(dists, labels)
        self.threshold = threshold
        self.threshold_sense = sense
        print(f"[AE] Computed Threshold: {threshold:.6f}, sense={sense}")
        return self.threshold, self.threshold_sense

    @staticmethod
    def _find_optimal_threshold(dists: np.ndarray, labels: np.ndarray) -> Tuple[float, str]:
        """Exhaustive search for optimal threshold and direction (ge/le)."""
        uniq = np.unique(dists)
        cands = [(uniq[i] + uniq[i + 1]) / 2.0 for i in range(len(uniq) - 1)]
        cands.insert(0, uniq[0] - 1e-6)
        cands.append(uniq[-1] + 1e-6)

        best_acc, best_thr, best_sense = -1.0, cands[0], "ge"
        for thr in cands:
            for sense in ("ge", "le"):
                preds = (dists >= thr).astype(int) if sense == "ge" else (dists <= thr).astype(int)
                acc = (preds == labels).mean()
                if acc > best_acc:
                    best_acc, best_thr, best_sense = acc, thr, sense

        return float(best_thr), best_sense

    @staticmethod
    def _parse_distance_value(val) -> float:
        """Parse distance value from CSV (handles scalars, lists, strings)."""
        if isinstance(val, (int, float, np.floating)):
            return float(val)
        s = str(val).strip()
        try:
            return float(s)
        except Exception:
            pass
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return float(v[0])
            return float(v)
        except Exception:
            pass
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", s)
        if m:
            return float(m.group(0))
        return np.nan

    @staticmethod
    def extract_distances_from_df(
        df, gen: str, limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract (distances, labels) for gen from official aeroblade DataFrame."""
        import pandas as pd

        mask = df["repo_id"].astype(str).str.strip() != "max"
        sub = df[mask]

        dists_list: List[float] = []
        labels_list: List[int] = []
        for _, row in sub.iterrows():
            dir_val = str(row.get("dir", ""))
            dist_val = AEDetector._parse_distance_value(row.get("distance", ""))
            if np.isnan(dist_val):
                continue
            if f"/{gen}/" in dir_val:
                dists_list.append(dist_val)
                labels_list.append(1)
            elif "/real/" in dir_val:
                dists_list.append(dist_val)
                labels_list.append(0)

        dists = np.array(dists_list, dtype=np.float64)
        labels = np.array(labels_list, dtype=np.int32)

        if limit and limit > 0:
            half = limit // 2
            real_idx = np.where(labels == 0)[0][:half]
            fake_idx = np.where(labels == 1)[0][:half]
            keep = np.concatenate([real_idx, fake_idx])
            dists = dists[keep]
            labels = labels[keep]

        return dists, labels

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict real/fake for images.

        Returns:
            predictions: [B] tensor of 0 (real) or 1 (fake)
            probabilities: [B] fake probability
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call find_threshold() first.")

        dists = self.compute_distances(images)

        if self.threshold_sense == "ge":
            preds = (dists >= self.threshold).long()
            probs = torch.sigmoid((dists - self.threshold) * 10)
        else:
            preds = (dists <= self.threshold).long()
            probs = torch.sigmoid((self.threshold - dists) * 10)

        return preds, probs

    @torch.no_grad()
    def evaluate_with_threshold(self, dataset: Dataset, threshold: float, sense: str, batch_size: int = 4, return_scores: bool = False):
        """Evaluate with specific threshold."""
        self.threshold = threshold
        self.threshold_sense = sense
        self.reset_generator()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        y_true, y_pred, y_scores = [], [], []
        for images, labels, _ in tqdm(loader, desc="AE eval", leave=False):
            dists = self.compute_distances(images)

            if sense == "ge":
                preds = (dists >= threshold).long()
            else:
                preds = (dists <= threshold).long()

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())
            if sense == "ge":
                y_scores.extend(dists.cpu().numpy())
            else:
                y_scores.extend(-dists.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        acc = (y_pred == y_true).mean() * 100

        real_mask = y_true == 0
        fake_mask = y_true == 1
        real_acc = (y_pred[real_mask] == 0).mean() * 100 if real_mask.sum() > 0 else 0.0
        fake_acc = (y_pred[fake_mask] == 1).mean() * 100 if fake_mask.sum() > 0 else 0.0
        print(f"    [detail] real_acc={real_acc:.2f}% ({real_mask.sum()} samples), fake_acc={fake_acc:.2f}% ({fake_mask.sum()} samples)")

        if return_scores:
            return acc, y_true, y_scores
        return acc

    @torch.no_grad()
    def cross_evaluate(
        self,
        data_root: Path,
        calib_csv_path: Path,
        gens: Optional[List[str]] = None,
        batch_size: int = 4,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Full cross-evaluation: calibration threshold + test distances + NxN matrix.

        Args:
            data_root: root dir containing real/test, <gen>/test, ...
            calib_csv_path: path to pre-computed calibration CSV
            gens: list of generator names (default: AE_GENS)
            batch_size: batch size for distance computation
            limit: max samples per dataset

        Returns:
            (acc_matrix, auc_matrix, roc_data)
        """
        import pandas as pd
        from sklearn.metrics import roc_auc_score

        if gens is None:
            gens = list(AE_GENS)

        data_root = Path(data_root)
        calib_csv_path = Path(calib_csv_path)

        if not calib_csv_path.exists():
            raise FileNotFoundError(
                f"Calibration CSV not found: {calib_csv_path}"
            )
        print(f"[AE] Loading calibration distances: {calib_csv_path}")
        calib_df = pd.read_csv(calib_csv_path)

        print("[AE] Computing test distances...")
        half_limit = limit // 2 if limit else None

        real_dists = self.compute_dir_distances(
            data_root / "real" / "test", batch_size=batch_size, limit=half_limit,
        )
        print(f"  real/test: {len(real_dists)} images")

        gen_dists: Dict[str, np.ndarray] = {}
        for gen in gens:
            gen_dists[gen] = self.compute_dir_distances(
                data_root / gen / "test", batch_size=batch_size, limit=half_limit,
            )
            print(f"  {gen}/test: {len(gen_dists[gen])} images")

        n = len(gens)
        mat_acc = np.zeros((n, n), dtype=np.float32)
        mat_auc = np.zeros((n, n), dtype=np.float32)
        roc_data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

        for i, train_gen in enumerate(gens):
            print(f"\n[AE] Threshold from: {train_gen}")

            calib_dists, calib_labels = self.extract_distances_from_df(
                calib_df, train_gen, limit=limit,
            )
            if len(calib_dists) == 0:
                print(f"  [skip] no calibration data for {train_gen}")
                continue

            threshold, sense = self._find_optimal_threshold(calib_dists, calib_labels)
            calib_preds = (calib_dists >= threshold).astype(int) if sense == "ge" else (calib_dists <= threshold).astype(int)
            calib_acc = (calib_preds == calib_labels).mean() * 100
            print(f"  Threshold={threshold:.6f}, sense={sense}, calib_acc={calib_acc:.2f}%")

            for j, test_gen in enumerate(gens):
                fake_d = gen_dists.get(test_gen, np.array([]))
                if len(real_dists) == 0 and len(fake_d) == 0:
                    continue

                test_d = np.concatenate([real_dists, fake_d])
                test_labels = np.array(
                    [0] * len(real_dists) + [1] * len(fake_d), dtype=np.int32,
                )

                if sense == "ge":
                    preds = (test_d >= threshold).astype(int)
                    scores = test_d.copy()
                else:
                    preds = (test_d <= threshold).astype(int)
                    scores = -test_d.copy()

                acc = (preds == test_labels).mean() * 100
                mat_acc[i, j] = acc
                roc_data[(train_gen, test_gen)] = (test_labels, scores)
                try:
                    auc = roc_auc_score(test_labels, scores)
                except ValueError:
                    auc = 0.5
                mat_auc[i, j] = auc

                print(f"  {train_gen} -> {test_gen}: ACC={acc:.2f}%, AUC={auc:.3f}")

        return mat_acc, mat_auc, roc_data


# ==============================================================================
# Factory
# ==============================================================================
def create_detector(detector_type: str, device: str = "cuda", **kwargs):
    """Create a detector by type.

    Args:
        detector_type: "dire", "lare", or "ae"/"aeroblade"
        device: "cuda" or "cpu"
        **kwargs: extra arguments passed to the detector constructor
    """
    if detector_type.lower() == "dire":
        return DIREDetector(device=device, **kwargs)
    elif detector_type.lower() == "lare":
        return LaREDetector(device=device, **kwargs)
    elif detector_type.lower() in ("ae", "aeroblade"):
        return AEDetector(device=device)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
