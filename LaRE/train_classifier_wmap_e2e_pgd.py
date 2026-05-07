#!/usr/bin/env python3
"""
LaRE E2E PGD Adversarial Training Script.

PGD (Projected Gradient Descent) adversarial training:
- L-inf norm constraint
- Default 8-step PGD (PGD-8)
- epsilon = 8/255, alpha = 2/255
- Each step updates delta using sign(grad)

Difference from FreeLB:
1. PGD uses sign(grad); FreeLB uses normalized grad
2. PGD optimizes independently each step; FreeLB accumulates gradients before update
3. PGD is more stable; FreeLB is theoretically stronger
"""

import json
import os
import cv2
import time
import shutil
import random
import datetime
import argparse
import numpy as np
import warnings
import logging as logger

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_

import albumentations as A
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve
import importlib
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate

# Diffusers for E2E LaRE extraction
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

try:
    import torchattacks
    TORCHATTACKS_AVAILABLE = True
except ImportError:
    TORCHATTACKS_AVAILABLE = False
    logger.warning("torchattacks not installed. Install with: pip install torchattacks")

test_best = -1
test_best_close = -1

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed: int) -> None:
    """Fix all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: int) -> None:
    """Ensure every DataLoader worker uses the same deterministic RNG."""
    np.random.seed(base_seed)
    random.seed(base_seed)


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)
        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# ==============================================================================
# LaRE Extractor (online loss map computation)
# ==============================================================================
class LareExtractor(nn.Module):
    """Differentiable LaRE loss map extractor based on Stable Diffusion."""

    def __init__(
        self,
        model_path: str,
        t: int = 100,
        ensemble_size: int = 1,
        sd_img_size: int = 512,
        prompt: str = "a photo",
        dtype: str = "fp32",
        seed: int = 42,
        device: torch.device = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t = t
        self.ensemble_size = ensemble_size
        self.sd_img_size = sd_img_size
        self.prompt = prompt
        self.base_seed = seed
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Load SD components
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(self.device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        if dtype == "fp16":
            self.unet.half()
            self.text_encoder.half()
            self.vae.half()

        self.dtype = torch.float16 if dtype == "fp16" else torch.float32

        # Gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if hasattr(self.vae, 'enable_gradient_checkpointing'):
                self.vae.enable_gradient_checkpointing()

        # Pre-compute text embeddings (fixed prompt)
        text_inputs = self.tokenizer(
            [self.prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            self.register_buffer("text_embeddings", self.text_encoder(text_inputs["input_ids"])[0])

        self.unet.eval()
        self.text_encoder.eval()
        self.vae.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute LaRE loss map.

        Args:
            images: [B, 3, H, W] in range [0, 1]

        Returns:
            loss_maps: [B, 4, 32, 32] (or other size depending on VAE)
        """
        bsz = images.shape[0]

        # Normalize to [-1, 1] and resize
        imgs_sd = (images * 2.0 - 1.0).to(self.dtype)
        if self.sd_img_size > 0:
            imgs_sd = F.interpolate(imgs_sd, size=(self.sd_img_size, self.sd_img_size),
                                   mode="bilinear", align_corners=False)

        # Fix random seed for determinism (supports gradient backprop)
        # DataParallel compatible: get current tensor device dynamically
        current_device = imgs_sd.device
        fork_devices = [current_device] if current_device.type == 'cuda' else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.base_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.base_seed)

            # VAE encode
            latents_dist = self.vae.encode(imgs_sd).latent_dist
            eps = torch.randn_like(latents_dist.mean)
            latents = latents_dist.mean + latents_dist.std * eps
            latents = latents * self.vae.config.scaling_factor

            # Expand for ensemble
            latents = latents.repeat_interleave(self.ensemble_size, dim=0)
            noise = torch.randn_like(latents)
            # DataParallel compatible: timesteps must be on same device as latents
            timesteps = torch.full((latents.shape[0],), self.t, device=latents.device, dtype=torch.long)
            # DataParallel compatible: scheduler's alphas_cumprod is not a buffer, move manually
            if self.noise_scheduler.alphas_cumprod.device != latents.device:
                self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(latents.device)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Expand text embeddings
            encoder_hidden_states = self.text_embeddings.repeat(latents.shape[0], 1, 1)

            # UNet prediction
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute target
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            # MSE loss map
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

        # Average over ensemble
        loss = loss.reshape(bsz, self.ensemble_size, *loss.shape[1:]).mean(dim=1)
        return loss  # [B, 4, 32, 32] or [B, 4, 64, 64]


# ==============================================================================
# Dataset
# ==============================================================================
class ImageDatasetE2E(Dataset):
    """Dataset for E2E training - does not require pre-computed loss maps."""

    def __init__(self, data_root, train_file, data_size=512, val_ratio=None,
                 split_anchor=True, args=None, max_samples=0):
        self.data_root = data_root
        self.data_size = data_size
        self.train_list = []
        self.anchor_list = []
        self.isAnchor = False
        self.isVal = False
        self.split_anchor = split_anchor

        self.albu_pre_train = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                A.ToGray(p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.HorizontalFlip(p=0.33),
        ], p=1.0)

        self.albu_pre_train_easy = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)

        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)

        self.imagenet_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # For adversarial training, return [0,1] range images; normalization is done inside the model
        ])

        self.args = args

        # Collect real (label=0) and fake (label=1) samples separately
        real_list = []
        fake_list = []

        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()

        while line:
            image_path, image_label = line.rsplit(' ', 1)
            label = int(image_label)
            if self.split_anchor and random.random() < 0.1 and label == 0 and len(self.anchor_list) < 100:
                self.anchor_list.append((image_path, label))
            else:
                if label == 0:
                    real_list.append((image_path, label))
                else:
                    fake_list.append((image_path, label))
            line = train_file_buf.readline().strip()

        # Limit samples per class (0 = unlimited)
        if max_samples > 0:
            real_list = real_list[:max_samples]
            fake_list = fake_list[:max_samples]

        self.train_list = real_list + fake_list

        if val_ratio is not None:
            np.random.shuffle(self.train_list)
            self.test_list = self.train_list[-int(len(self.train_list) * val_ratio):]
            self.train_list = self.train_list[:-int(len(self.train_list) * val_ratio)]
        else:
            self.test_list = self.train_list

    def transform(self, x):
        if self.isVal:
            x = self.albu_pre_val(image=x)['image']
        else:
            if self.args and self.args.no_strong_aug:
                x = self.albu_pre_train_easy(image=x)['image']
            else:
                x = self.albu_pre_train(image=x)['image']
        x = self.imagenet_norm(x)
        return x

    def __len__(self):
        if self.isAnchor:
            return len(self.anchor_list)
        elif self.isVal:
            return len(self.test_list)
        else:
            return len(self.train_list)

    def __getitem__(self, index):
        if self.isAnchor:
            return self.getitem(index, self.anchor_list)
        elif self.isVal:
            return self.getitem(index, self.test_list)
        else:
            return self.getitem(index, self.train_list)

    def getitem(self, index, data_list):
        image_path, onehot_label = data_list[index]

        if not os.path.exists(image_path):
            image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)

        if image is None:
            logger.info('Error Image: %s' % image_path)
            image = np.zeros([512, 512, 3], dtype=np.uint8)
        image = image[..., ::-1]

        crop = self.transform(image)
        onehot_label = torch.LongTensor([onehot_label])
        return crop, onehot_label

    def set_val_True(self):
        self.isVal = True

    def set_val_False(self):
        self.isVal = False

    def set_anchor_True(self):
        self.isAnchor = True

    def set_anchor_False(self):
        self.isAnchor = False


# ==============================================================================
# PGD Adversarial Training
# ==============================================================================
def normalize_for_clip(images, device):
    """ImageNet normalization."""
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (images - mean) / std


# ==============================================================================
# Torchattacks integration
# ==============================================================================
class LaREModelWrapper(nn.Module):
    """
    Wrapper for torchattacks integration.
    Wraps model + lare_extractor as a single unit for torchattacks.

    Two modes:
    - approx: loss_map does not backprop through SD (saves memory)
    - full: loss_map backprops through SD (full E2E)
    """

    def __init__(self, model, lare_extractor, device, freeze_lare=True):
        super().__init__()
        self.model = model
        self.lare_extractor = lare_extractor
        self.device = device
        self.freeze_lare = freeze_lare

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] in [0, 1] range

        Returns:
            logits: [B, num_classes]
        """
        if self.freeze_lare:
            with torch.no_grad():
                loss_maps = self.lare_extractor(x)
        else:
            loss_maps = self.lare_extractor(x)

        x_norm = (x - self.imagenet_mean.to(x.device)) / self.imagenet_std.to(x.device)
        logits = self.model(x_norm, loss_maps)
        return logits


def _train_one_epoch_with_torchattacks(
    data_loader, model, lare_extractor, optimizer, cur_epoch,
    loss_meter, args, device, writer, ngpus_per_node,
    freeze_lare=True, mode_name="Approx"
):
    """
    PGD adversarial training using the torchattacks library.

    Args:
        freeze_lare: True=approx mode (no SD backprop), False=full mode (full E2E)
        mode_name: mode name for logging
    """
    if not TORCHATTACKS_AVAILABLE:
        raise ImportError("torchattacks is not installed. Run: pip install torchattacks")

    loss_meter.reset()
    batch_idx = 0
    global_step = 0

    wrapper = LaREModelWrapper(model, lare_extractor, device, freeze_lare=freeze_lare)

    atk = torchattacks.PGD(
        wrapper,
        eps=args.pgd_eps,
        alpha=args.pgd_alpha,
        steps=args.pgd_k,
        random_start=getattr(args, 'pgd_rand_init', True)
    )

    pbar = tqdm(data_loader, desc=f'Epoch {cur_epoch:03d} [PGD-{mode_name}-TA]', dynamic_ncols=True)

    clean_acc_meter = AverageMeter()
    adv_acc_meter = AverageMeter()

    for (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device).flatten()

        # Generate adversarial examples
        wrapper.eval()
        adv_images = atk(images, labels)
        wrapper.train()

        # Clean accuracy
        with torch.no_grad():
            clean_logits = wrapper(images)
            clean_pred = clean_logits.argmax(dim=1)
            clean_batch_acc = (clean_pred == labels).float().mean().item()
            clean_acc_meter.update(clean_batch_acc, images.shape[0])

        # Adversarial training step
        adv_logits = wrapper(adv_images)
        loss = args.criterion_ce(adv_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])

        # Adversarial accuracy
        with torch.no_grad():
            adv_pred = adv_logits.argmax(dim=1)
            adv_batch_acc = (adv_pred == labels).float().mean().item()
            adv_acc_meter.update(adv_batch_acc, images.shape[0])

        global_step += 1

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'c_acc': f'{clean_acc_meter.avg:.4f}',
            'a_acc': f'{adv_acc_meter.avg:.4f}',
            'lr': f'{get_lr(optimizer):.2e}'
        })

        if (not args.multiprocessing_distributed or
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) \
            and batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                f'Ep {cur_epoch:03d}, it {batch_idx:03d}/{len(data_loader)}, '
                f'lr: {lr:.7f}, CE: {loss_avg:.6f}, '
                f'C_ACC: {clean_acc_meter.avg:.4f}, A_ACC: {adv_acc_meter.avg:.4f} '
                f'[PGD-{mode_name}-Torchattacks]'
            )
            loss_meter.reset()
            writer.add_scalar('train/loss', loss_avg, global_step)
            writer.add_scalar('train/clean_acc', clean_acc_meter.avg, global_step)
            writer.add_scalar('train/adv_acc', adv_acc_meter.avg, global_step)
            writer.add_scalar('train/lr', lr, global_step)

        if args.break_onek and batch_idx > 1000:
            break
        batch_idx += 1

    logger.info(f'End Training (PGD-{mode_name}-Torchattacks) - '
                f'Clean ACC: {clean_acc_meter.avg:.4f}, ADV ACC: {adv_acc_meter.avg:.4f}')
    return loss_meter.avg


def train_one_epoch_pgd_approx(
    data_loader, model, lare_extractor, optimizer, cur_epoch,
    loss_meter, args, device, writer, ngpus_per_node
):
    """
    Approximate E2E PGD adversarial training (recommended).

    PGD characteristics:
    1. Updates delta using sign(grad) (not normalized grad)
    2. Updates model parameters independently each step (no gradient accumulation)
    3. L-inf norm constraint

    "Approximate" means:
    - Inner loop recomputes loss_map each step (tracking adversarial image), but does not
      backprop through SD (saves memory)
    - Only optimizes classifier parameters and adversarial perturbation delta
    """
    if getattr(args, 'use_torchattacks', False):
        return _train_one_epoch_with_torchattacks(
            data_loader, model, lare_extractor, optimizer, cur_epoch,
            loss_meter, args, device, writer, ngpus_per_node,
            freeze_lare=True, mode_name="Approx"
        )

    loss_meter.reset()
    batch_idx = 0
    global_step = 0

    eps = args.pgd_eps
    alpha = args.pgd_alpha
    K = args.pgd_k

    pbar = tqdm(data_loader, desc=f'Epoch {cur_epoch:03d} [PGD-Approx]', dynamic_ncols=True)

    acc_meter = AverageMeter()

    for (images, labels) in pbar:
        images = images.to(device)  # [B, 3, H, W] in [0, 1]
        labels = labels.to(device).flatten()
        B = images.shape[0]

        # Step 1: PGD inner loop
        if getattr(args, 'pgd_rand_init', True):
            delta = torch.zeros_like(images).uniform_(-eps, eps)
        else:
            delta = torch.zeros_like(images)
        delta = delta.detach()

        # PGD inner loop - only update delta, model params frozen
        for k in range(K):
            delta.requires_grad_(True)
            adv_images = torch.clamp(images + delta, 0, 1)

            # Recompute loss_map each step (tracking adversarial image), no SD backprop
            with torch.no_grad():
                adv_loss_maps = lare_extractor(adv_images)

            adv_images_norm = normalize_for_clip(adv_images, device)
            logits = model(adv_images_norm, adv_loss_maps)

            # Compute loss, gradient only w.r.t. delta (no model gradient accumulation)
            loss = args.criterion_ce(logits, labels)
            grad = torch.autograd.grad(loss, delta)[0]

            # PGD: sign(grad) update
            delta = delta.detach() + alpha * grad.sign()
            delta = torch.clamp(delta, -eps, eps)
            delta = torch.clamp(images + delta, 0, 1) - images
            delta = delta.detach()

        # Step 2: Final forward with worst-case delta, update model once
        adv_images = torch.clamp(images + delta.detach(), 0, 1)
        with torch.no_grad():
            adv_loss_maps = lare_extractor(adv_images)
        adv_images_norm = normalize_for_clip(adv_images, device)
        logits = model(adv_images_norm, adv_loss_maps)
        loss = args.criterion_ce(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            batch_acc = (pred == labels).float().mean().item()
            acc_meter.update(batch_acc, images.shape[0])

        global_step += 1

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
            'lr': f'{get_lr(optimizer):.2e}'
        })

        if (not args.multiprocessing_distributed or
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) \
            and batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                'Ep %03d, it %03d/%03d, lr: %8.7f, CE: %7.6f [PGD-Approx]' %
                (cur_epoch, batch_idx, len(data_loader), lr, loss_avg)
            )
            loss_meter.reset()
            writer.add_scalar('train/loss', loss_avg, global_step)
            writer.add_scalar('train/lr', lr, global_step)

        if args.break_onek and batch_idx > 1000:
            break
        batch_idx += 1

    logger.info('End Training (PGD-Approx)')
    return loss_meter.avg


def train_one_epoch_pgd_full(
    data_loader, model, lare_extractor, optimizer, cur_epoch,
    loss_meter, args, device, writer, ngpus_per_node
):
    """
    Full E2E PGD adversarial training.

    Recomputes loss_map every inner loop step; gradients flow through classifier parameters.

    Warning: requires significant GPU memory. Recommended settings:
    - Use gradient checkpointing
    - Reduce batch size
    - Use fp16
    """
    if getattr(args, 'use_torchattacks', False):
        return _train_one_epoch_with_torchattacks(
            data_loader, model, lare_extractor, optimizer, cur_epoch,
            loss_meter, args, device, writer, ngpus_per_node,
            freeze_lare=False, mode_name="Full"
        )

    loss_meter.reset()
    batch_idx = 0
    global_step = 0

    eps = args.pgd_eps
    alpha = args.pgd_alpha
    K = args.pgd_k

    pbar = tqdm(data_loader, desc=f'Epoch {cur_epoch:03d} [PGD-Full]', dynamic_ncols=True)

    acc_meter = AverageMeter()

    for (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device).flatten()
        B = images.shape[0]

        # Random init delta
        if getattr(args, 'pgd_rand_init', True):
            delta = torch.zeros_like(images).uniform_(-eps, eps)
        else:
            delta = torch.zeros_like(images)
        delta = delta.detach()

        for k in range(K):
            delta.requires_grad_(True)
            adv_images = torch.clamp(images + delta, 0, 1)

            # Recompute loss_map each step
            loss_maps = lare_extractor(adv_images)

            adv_images_norm = normalize_for_clip(adv_images, device)
            logits = model(adv_images_norm, loss_maps)

            loss = args.criterion_ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            # PGD: sign(grad) update
            if delta.grad is not None:
                delta_grad = delta.grad.data
                delta = delta.detach() + alpha * delta_grad.sign()
                delta = torch.clamp(delta, -eps, eps)
                delta = torch.clamp(images + delta, 0, 1) - images
                delta = delta.detach()
            else:
                delta = delta.detach()
                logger.warning(f"Delta grad is None at step {k}/{K}")

            # Update model parameters each step
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            batch_acc = (pred == labels).float().mean().item()
            acc_meter.update(batch_acc, images.shape[0])

        global_step += 1

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
            'lr': f'{get_lr(optimizer):.2e}'
        })

        if (not args.multiprocessing_distributed or
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) \
            and batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                'Ep %03d, it %03d/%03d, lr: %8.7f, CE: %7.6f [PGD-Full E2E]' %
                (cur_epoch, batch_idx, len(data_loader), lr, loss_avg)
            )
            loss_meter.reset()
            writer.add_scalar('train/loss', loss_avg, global_step)
            writer.add_scalar('train/lr', lr, global_step)

        if args.break_onek and batch_idx > 1000:
            break
        batch_idx += 1

    logger.info('End Training (PGD-Full E2E)')
    return loss_meter.avg


def train_one_epoch_clean(
    data_loader, model, lare_extractor, optimizer, cur_epoch,
    loss_meter, args, device, writer, ngpus_per_node
):
    """Standard clean training (no adversarial perturbation)."""
    loss_meter.reset()
    batch_idx = 0
    global_step = 0

    pbar = tqdm(data_loader, desc=f'Epoch {cur_epoch:03d} [Clean]', dynamic_ncols=True)

    acc_meter = AverageMeter()

    for (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device).flatten()

        with torch.no_grad():
            loss_maps = lare_extractor(images)

        images_norm = normalize_for_clip(images, device)
        logits = model(images_norm, loss_maps)
        loss = args.criterion_ce(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            batch_acc = (pred == labels).float().mean().item()
            acc_meter.update(batch_acc, images.shape[0])

        global_step += 1

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
            'lr': f'{get_lr(optimizer):.2e}'
        })

        if (not args.multiprocessing_distributed or
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) \
            and batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                'Ep %03d, it %03d/%03d, lr: %8.7f, CE: %7.6f [Clean]' %
                (cur_epoch, batch_idx, len(data_loader), lr, loss_avg)
            )
            loss_meter.reset()
            writer.add_scalar('train/loss', loss_avg, global_step)
            writer.add_scalar('train/lr', lr, global_step)

        if args.break_onek and batch_idx > 1000:
            break
        batch_idx += 1

    logger.info('End Training (Clean)')
    return loss_meter.avg


def train_one_epoch_pgd_joint(
    data_loader, model, lare_extractor, optimizer, cur_epoch,
    loss_meter, args, device, writer, ngpus_per_node
):
    """
    Joint PGD adversarial training.

    D is an independent tensor; perturbations are added separately to x and LaRE features.
    Gradients flow to both classifier parameters and D.
    """
    loss_meter.reset()
    batch_idx = 0
    global_step = 0

    eps = args.pgd_eps
    alpha = args.pgd_alpha
    K = args.pgd_k

    pbar = tqdm(data_loader, desc=f'Epoch {cur_epoch:03d} [PGD-Joint]', dynamic_ncols=True)

    acc_meter = AverageMeter()

    for (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device).flatten()
        B = images.shape[0]

        # Initialize delta_x and delta_d
        delta_x = torch.zeros_like(images).uniform_(-eps, eps).detach()
        delta_d = torch.zeros_like(images).uniform_(-eps, eps).detach()

        for k in range(K):
            delta_x.requires_grad_(True)
            delta_d.requires_grad_(True)

            adv_images = torch.clamp(images + delta_x, 0, 1)
            adv_d = torch.clamp(delta_d, -eps, eps)

            loss_maps = lare_extractor(adv_images + adv_d)

            adv_images_norm = normalize_for_clip(adv_images, device)
            logits = model(adv_images_norm, loss_maps)

            loss = args.criterion_ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            if delta_x.grad is not None:
                delta_x = delta_x.detach() + alpha * delta_x.grad.data.sign()
                delta_x = torch.clamp(delta_x, -eps, eps)
                delta_x = torch.clamp(images + delta_x, 0, 1) - images

            if delta_d.grad is not None:
                delta_d = delta_d.detach() + alpha * delta_d.grad.data.sign()
                delta_d = torch.clamp(delta_d, -eps, eps)

            delta_x = delta_x.detach()
            delta_d = delta_d.detach()

            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == labels).sum().item()
            acc_meter.update(correct / max(images.shape[0], 1), images.shape[0])

        global_step += 1

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
            'lr': f'{get_lr(optimizer):.2e}'
        })

        if (not args.multiprocessing_distributed or
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) \
            and batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                'Ep %03d, it %03d/%03d, lr: %8.7f, CE: %7.6f [PGD-Joint]' %
                (cur_epoch, batch_idx, len(data_loader), lr, loss_avg)
            )
            loss_meter.reset()
            writer.add_scalar('train/loss', loss_avg, global_step)
            writer.add_scalar('train/lr', lr, global_step)

        if args.break_onek and batch_idx > 1000:
            break
        batch_idx += 1

    logger.info('End Training (PGD-Joint)')
    return loss_meter.avg


# ==============================================================================
# Validation
# ==============================================================================
def validation_e2e(model, lare_extractor, args, test_file, device, ngpus_per_node):
    """E2E validation."""
    logger.info('Start eval (E2E)')
    model.eval()

    val_dataset = ImageDatasetE2E(args.data_root, test_file, data_size=args.data_size,
                                  split_anchor=False, args=args,
                                  max_samples=args.max_val_samples)

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None

    worker_init = (lambda wid: seed_worker(wid, args.seed)) if args.seed is not None else None
    eval_generator = None
    if args.seed is not None:
        eval_generator = torch.Generator()
        eval_generator.manual_seed(args.seed)

    data_loader = DataLoader(
        val_dataset, args.test_batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        worker_init_fn=worker_init, generator=eval_generator
    )
    data_loader.dataset.set_val_True()

    gt_labels_list = []
    prob_labels_list = []

    with torch.no_grad():
        for iter, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.flatten().to(device)

            loss_maps = lare_extractor(images)

            images_norm = normalize_for_clip(images, device)
            logits = model(images_norm, loss_maps)
            prob = torch.softmax(logits, dim=-1)

            gt_labels_list.append(labels)
            prob_labels_list.append(prob[:, 1])

            if (not args.multiprocessing_distributed or
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) \
                and iter % 50 == 0 and iter > 0:
                logger.info('Eval: it %03d/%03d' % (iter, len(data_loader)))

    gt_labels = torch.cat(gt_labels_list, dim=0)
    prob_labels = torch.cat(prob_labels_list, dim=0)

    # Distributed training: aggregate results from all GPUs
    if args.distributed:
        gt_labels_gather = [torch.zeros_like(gt_labels) for _ in range(dist.get_world_size())]
        prob_labels_gather = [torch.zeros_like(prob_labels) for _ in range(dist.get_world_size())]

        dist.all_gather(gt_labels_gather, gt_labels)
        dist.all_gather(prob_labels_gather, prob_labels)

        gt_labels = torch.cat(gt_labels_gather, dim=0)
        prob_labels = torch.cat(prob_labels_gather, dim=0)

    gt_labels = gt_labels.cpu().numpy()
    prob_labels = prob_labels.cpu().numpy()

    # Compute metrics
    fpr, tpr, thres = roc_curve(gt_labels, prob_labels)
    precision, recall, thresholds = precision_recall_curve(gt_labels, prob_labels)
    f_score = precision * recall / (precision + recall + 1e-8)

    auc = roc_auc_score(gt_labels, prob_labels)
    ap = average_precision_score(gt_labels, prob_labels)

    # Search best threshold
    best_acc, best_thresh = search_best_acc(gt_labels, prob_labels)

    r_acc = accuracy_score(gt_labels[gt_labels == 0], prob_labels[gt_labels == 0] > 0.5)
    f_acc = accuracy_score(gt_labels[gt_labels == 1], prob_labels[gt_labels == 1] > 0.5)
    raw_acc = accuracy_score(gt_labels, prob_labels > 0.5)

    model.train()
    return auc, best_acc, ap, raw_acc, r_acc, f_acc


def search_best_acc(gt_labels, pred_probs):
    best_acc = -1
    best_threshold = -1
    for thresh in sorted(set(pred_probs.tolist())):
        pred = (pred_probs > thresh).astype(int)
        acc = accuracy_score(gt_labels, pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    return best_acc, best_threshold


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class FakeWriter:
    def add_scalar(self, p1, p2, p3):
        pass


# ==============================================================================
# Main
# ==============================================================================
def main(gpu, ngpus_per_node, args):
    global test_best
    global test_best_close

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir))
    else:
        writer = FakeWriter()

    args.gpu = gpu
    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Setup device
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device('cuda:{}'.format(args.gpu))
            torch.cuda.set_device(args.gpu)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load classifier model
    model = getattr(importlib.import_module('model'), args.model)(
        num_class=args.num_class, clip_type=args.clip_type
    )
    model = model.to(device)

    # Multi-GPU: DataParallel (single machine, simple)
    if args.gpu is None and torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    elif args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    # Load LaRE Extractor
    logger.info(f"Loading LaRE extractor from {args.sd_model_path}")
    lare_extractor = LareExtractor(
        model_path=args.sd_model_path,
        t=args.lare_t,
        ensemble_size=args.lare_ensemble,
        sd_img_size=args.sd_img_size,
        prompt=args.lare_prompt,
        dtype=args.lare_dtype,
        seed=args.seed or 42,
        device=device,
        use_gradient_checkpointing=(args.pgd_mode == "full"),
    )
    lare_extractor = lare_extractor.to(device)
    lare_extractor.eval()

    # Freeze extractor parameters
    for param in lare_extractor.parameters():
        param.requires_grad = False

    # Multi-GPU: DataParallel for LaRE extractor (memory balancing)
    if args.gpu is None and torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel for LaRE extractor on {torch.cuda.device_count()} GPUs")
        lare_extractor = torch.nn.DataParallel(lare_extractor)

    # Data loading
    train_dataset = ImageDatasetE2E(
        args.data_root, args.train_file, data_size=args.data_size,
        val_ratio=None, split_anchor=False, args=args,
        max_samples=args.max_train_samples
    )

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        logger.info(f'Train dataset size: {len(train_dataset)}')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    worker_init = (lambda wid: seed_worker(wid, args.seed)) if args.seed is not None else None
    train_generator = None
    if args.seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        worker_init_fn=worker_init, generator=train_generator
    )

    # Loss function
    if not args.label_smooth:
        args.criterion_ce = nn.CrossEntropyLoss().to(device)
    else:
        args.criterion_ce = LabelSmoothingLoss(classes=args.num_class, smoothing=args.smoothing)

    # Optimizer
    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded checkpoint from {args.resume}")

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.lr)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    loss_meter = AverageMeter()
    best_val_score = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    patience = getattr(args, "early_stop_patience", 10)

    if args.use_torchattacks and not TORCHATTACKS_AVAILABLE:
        raise ImportError("torchattacks is not installed. Run: pip install torchattacks")

    if args.pgd_mode == "approx":
        train_fn = train_one_epoch_pgd_approx
        impl = "Torchattacks" if args.use_torchattacks else "Manual"
        logger.info(f"Using PGD-Approx ({impl}): eps={args.pgd_eps}, alpha={args.pgd_alpha}, K={args.pgd_k}")
    elif args.pgd_mode == "full":
        train_fn = train_one_epoch_pgd_full
        impl = "Torchattacks" if args.use_torchattacks else "Manual"
        logger.info(f"Using PGD-Full E2E ({impl}): eps={args.pgd_eps}, alpha={args.pgd_alpha}, K={args.pgd_k}")
        logger.warning("Full E2E mode requires significant GPU memory!")
    elif args.pgd_mode == "joint":
        train_fn = train_one_epoch_pgd_joint
        impl = "Manual" if not args.use_torchattacks else "Torchattacks (not supported for joint)"
        logger.info(f"Using PGD-Joint mode ({impl}): eps={args.pgd_eps}, alpha={args.pgd_alpha}, K={args.pgd_k}")
        if args.use_torchattacks:
            logger.warning("Torchattacks not supported for joint mode, using manual PGD")
    else:
        train_fn = train_one_epoch_clean
        logger.info("Using Clean training (no adversarial)")

    for epoch in range(args.epoches):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        train_data_loader.dataset.set_val_False()

        train_fn(train_data_loader, model, lare_extractor, optimizer, epoch,
                 loss_meter, args, device, writer, ngpus_per_node)

        # Save latest model
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'latest.pt'))

        # Validation
        val_auc, val_acc, val_ap, val_raw_acc, val_r_acc, val_f_acc = validation_e2e(
            model, lare_extractor, args, args.val_file, device, ngpus_per_node
        )
        val_score = val_auc

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            logger.info(
                f'Epoch {epoch}: Val AUC: {val_auc:.4f}, ACC: {val_acc:.4f}, AP: {val_ap:.4f}, '
                f'Raw ACC: {val_raw_acc:.4f}, Real ACC: {val_r_acc:.4f}, Fake ACC: {val_f_acc:.4f}'
            )
            writer.add_scalar('val/AUC', val_auc, epoch)
            writer.add_scalar('val/ACC', val_acc, epoch)

            if val_acc > test_best_close:
                test_best_close = val_acc
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'Val_best.pth'))
                logger.info(f'Epoch: {epoch}, Best Val (by ACC)')

        # Early stopping
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'earlystop_best.pth'))
        else:
            no_improve_epochs += 1

        if patience > 0 and no_improve_epochs >= patience:
            logger.info(f'Early stopping at epoch {epoch}, best epoch {best_epoch}')
            break

        lr_schedule.step(val_score)

    logger.info("Training finished!")


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description="LaRE E2E PGD Adversarial Training")

    # Data arguments
    conf.add_argument("--data_root", type=str, default='data/')
    conf.add_argument("--train_file", type=str, required=True)
    conf.add_argument("--val_file", type=str, required=True)
    conf.add_argument("--test_file", type=str, default='')
    conf.add_argument('--data_size', type=int, default=224)

    # Model arguments
    conf.add_argument("--model", type=str, default='CLipClassifierWMapV5')
    conf.add_argument("--num_class", type=int, default=2)
    conf.add_argument("--clip_type", type=str, default='RN50x64')

    # LaRE Extractor arguments
    conf.add_argument("--sd_model_path", type=str,
                      default=os.environ.get("SD21_MODEL_PATH", "/path/to/stable-diffusion-2-1-base"),
                      help="Path to SD 2.1-base model dir (or set SD21_MODEL_PATH env var)")
    conf.add_argument("--lare_t", type=int, default=100, help="Diffusion timestep for LaRE")
    conf.add_argument("--lare_ensemble", type=int, default=1, help="Ensemble size for LaRE")
    conf.add_argument("--sd_img_size", type=int, default=512, help="Image size for SD")
    conf.add_argument("--lare_prompt", type=str, default="a photo")
    conf.add_argument("--lare_dtype", type=str, default="fp32", choices=["fp16", "fp32"])

    # PGD arguments
    conf.add_argument("--pgd_mode", type=str, default="approx",
                      choices=["approx", "full", "clean", "joint"],
                      help="approx: approximate E2E (recommended), full: full E2E (high memory), clean: no adversarial")
    conf.add_argument("--use_torchattacks", action='store_true', default=False,
                      help="Use torchattacks library instead of manual PGD (approx/full modes only)")
    conf.add_argument("--pgd_eps", type=float, default=8/255, help="Perturbation budget (8/255)")
    conf.add_argument("--pgd_alpha", type=float, default=2/255, help="Step size (2/255)")
    conf.add_argument("--pgd_k", type=int, default=8, help="Number of PGD steps (default: 8)")
    conf.add_argument("--pgd_rand_init", action='store_true', default=True,
                      help="Random initialization for PGD (recommended)")

    # Training arguments
    conf.add_argument('--lr', type=float, default=1e-4)
    conf.add_argument('--epoches', type=int, default=100)
    conf.add_argument('--batch_size', type=int, default=8, help="Reduce for E2E mode")
    conf.add_argument('--test_batch_size', type=int, default=8)
    conf.add_argument("--resume", type=str, default='')
    conf.add_argument("--out_dir", type=str, required=True)
    conf.add_argument("--break_onek", action='store_true', default=False)
    conf.add_argument("--no_strong_aug", action='store_true', default=False)
    conf.add_argument("--label_smooth", action='store_true', default=False)
    conf.add_argument('--smoothing', type=float, default=0.1)
    conf.add_argument("--seed", type=int, default=42)
    conf.add_argument("--early_stop_patience", type=int, default=10)
    conf.add_argument("--max_train_samples", type=int, default=0,
                      help="Max training samples per class (0=unlimited)")
    conf.add_argument("--max_val_samples", type=int, default=0,
                      help="Max validation samples per class (0=unlimited)")

    # Distributed arguments
    conf.add_argument("--gpu", type=int, default=None)
    conf.add_argument('--multiprocessing-distributed', action='store_true')
    conf.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str)
    conf.add_argument('--dist-backend', default='nccl', type=str)
    conf.add_argument('--world-size', default=-1, type=int)
    conf.add_argument('--rank', default=-1, type=int)
    conf.add_argument('-j', '--workers', default=4, type=int)

    args = conf.parse_args()

    logger.info(args)

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    # Create output directory
    date_now = datetime.datetime.now()
    date_str = 'E2E_PGD_%s_Log_v%02d%02d%02d%02d' % (
        args.pgd_mode, date_now.month, date_now.day, date_now.hour, date_now.minute
    )
    args.out_dir = os.path.join(args.out_dir, date_str)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(args.gpu, 1, args)
