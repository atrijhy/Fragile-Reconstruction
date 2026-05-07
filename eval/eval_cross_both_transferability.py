#!/usr/bin/env python3
"""
Cross-Both Adversarial Transferability Evaluation.

Tests dual transferability: across both detector methods and generator-specific classifiers.
Heatmap (i, j): adversarial examples crafted by Attack_Method against cls_i on data_j,
evaluated on Eval_Method's cls_j.

Produces 6 heatmaps (all cross-method pairs):
    DIRE -> LaRE², DIRE -> AeroBlade
    LaRE² -> DIRE, LaRE² -> AeroBlade
    AeroBlade -> DIRE, AeroBlade -> LaRE²

Set PYTHONPATH to include DIRE/, LaRE/, aeroblade/src/, DiffPure/ before running.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from shared_detectors import DIREDetector, LaREDetector, AEDetector

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GENS = ["adm", "flux", "sdv5", "vqdm"]
GEN_DISPLAY = {"adm": "ADM", "flux": "FLUX", "sdv5": "SD1.5", "vqdm": "VQDM"}

METHODS = ["dire", "lare", "aeroblade"]
METHOD_DISPLAY = {"dire": "DIRE", "lare": r"LaRE$^2$", "aeroblade": "AEROBLADE"}

# Paths configured via argparse (see main())
DIRE_ADV_ROOT = None
LARE_ADV_ROOT = None
AE_ADV_ROOT = None
DIRE_VALID_INDICES_DIR = None
DIRE_CLEAN_RIGHT_ONLY_DIR = None

USE_CLEAN_RIGHT_ONLY = False
USE_ALL_SAMPLES = False

def load_dire_valid_indices(attack_gen: str, data_gen: str, is_fake: bool) -> Optional[set]:
    """Load DIRE valid indices from pre-extracted index files.
    
    Uses either:
    - clean_right AND attack_success (default, USE_CLEAN_RIGHT_ONLY=False)
    - clean_right only (USE_CLEAN_RIGHT_ONLY=True)
    - all samples (USE_ALL_SAMPLES=True)
    
    Args:
        attack_gen: Attack classifier's training dataset (adm, flux, sdv5, vqdm)
        data_gen: Dataset being attacked (adm, flux, sdv5, vqdm)
        is_fake: True for fake images, False for real images
        
    Returns:
        Set of valid sample indices to include, or None if file not found.
    """
    if USE_ALL_SAMPLES:
        return None  # Include all samples without filtering

    if is_fake:
        filename = f"fake_cls_{attack_gen}_on_{data_gen}.txt"
    else:
        filename = f"real_cls_{attack_gen}.txt"
    
    # Choose index directory based on global flags
    if USE_CLEAN_RIGHT_ONLY:
        index_dir = DIRE_CLEAN_RIGHT_ONLY_DIR
    else:
        index_dir = DIRE_VALID_INDICES_DIR

    filepath = index_dir / filename
    if not filepath.exists():
        return None
    
    indices = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                indices.add(int(line))
    return indices


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# ECCV-style Heatmap
# ==============================================================================
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':   'stix',
    'font.size':          11,
    'axes.labelsize':     12.5,
    'axes.titlesize':     14,
    'xtick.labelsize':    11.5,
    'ytick.labelsize':    11.5,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.08,
})


def draw_single_heatmap(matrix, title, vmin, vmax, cmap, filename_base, out_dir,
                        row_labels=None, col_labels=None,
                        xlabel='Eval Classifier', ylabel='Attack Classifier'):
    """Draw a single annotated heatmap as separate figure."""
    if row_labels is None:
        row_labels = [GEN_DISPLAY.get(g, g) for g in GENS]
    if col_labels is None:
        col_labels = [GEN_DISPLAY.get(g, g) for g in GENS]
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    n_row, n_col = matrix.shape
    ax.set_xticks(np.arange(n_col))
    ax.set_yticks(np.arange(n_row))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Put x-labels on bottom
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Annotate cells
    thresh_hi = vmin + (vmax - vmin) * 0.65
    for i in range(n_row):
        for j in range(n_col):
            val = matrix[i, j]
            color = 'white' if val > thresh_hi else '#222222'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)

    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Minor ticks for grid lines
    ax.set_xticks(np.arange(n_col + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_row + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=25, pad=0.025)
    cbar.set_label('Accuracy (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{filename_base}.pdf')
    fig.savefig(out_dir / f'{filename_base}.png')
    plt.close(fig)
    print(f'Saved: {filename_base}')


# ==============================================================================
# Adversarial Image Dataset
# ==============================================================================
class AdvImageDataset(Dataset):
    """Dataset for loading adversarial images from attack directories."""

    def __init__(self, img_dirs, data_size: int = 256, limit: int = None, label: int = 1, 
                 include_indices: Optional[set] = None):
        """
        Args:
            img_dirs: Directory or list of directories containing adversarial images
            data_size: Target image size
            limit: Max number of samples to load
            label: Label for all samples (0=real, 1=fake)
            include_indices: Set of sample indices to include (for DIRE valid samples only)
        """
        self.data_size = data_size
        self.label = label
        self.samples: List[Path] = []
        
        # Support both single Path and list of Paths
        if isinstance(img_dirs, Path):
            img_dirs = [img_dirs]
        
        # Collect all images from all directories
        for img_dir in img_dirs:
            if not img_dir.exists():
                continue
            for ext in ["*.png", "*.PNG", "*.jpg", "*.jpeg"]:
                self.samples.extend(img_dir.glob(ext))
        self.samples = sorted(set(self.samples))
        
        # Filter to include only valid samples by index (for DIRE)
        # DIRE files are named sample_XXXXXX_adv.png, extract index from filename
        if include_indices is not None:
            original_count = len(self.samples)
            filtered_samples = []
            for s in self.samples:
                # Try to extract index from filename like sample_000020_adv.png
                match = re.search(r'sample_(\d+)_adv', s.stem)
                if match:
                    idx = int(match.group(1))
                    if idx in include_indices:
                        filtered_samples.append(s)
                # Non-matching files are skipped when include_indices is specified
            self.samples = filtered_samples
            print(f"  [include] {original_count} -> {len(self.samples)} valid samples")
        
        if limit and limit > 0:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.data_size, self.data_size))
        tensor = TF.to_tensor(img)
        _, h, w = tensor.shape
        pad_h = max(self.data_size - h, 0)
        pad_w = max(self.data_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            tensor = TF.pad(tensor, padding, fill=0)
        tensor = TF.center_crop(tensor, [self.data_size, self.data_size])
        return tensor, self.label, str(img_path.stem)


# ==============================================================================
# Get adversarial image paths for each attack method
# ==============================================================================
def get_dire_adv_paths(attack_gen: str, data_gen: str) -> Tuple[Optional[Path], Optional[Path], Optional[set], Optional[set]]:
    """Get DIRE adversarial image paths.
    
    Returns (fake_adv_dir, real_adv_dir, fake_include_indices, real_include_indices).
    - fake: adversarial examples attacking cls_{attack_gen} on fake images from data_{data_gen}
    - real: adversarial examples attacking cls_{attack_gen} on real images
    - fake_include_indices/real_include_indices: Valid sample indices (clean_right AND attack_success)
    """
    fake_base = DIRE_ADV_ROOT / f"fake_cls_{attack_gen}_on_{data_gen}" / "dire_resnet50"
    real_base = DIRE_ADV_ROOT / f"real_cls_{attack_gen}" / "dire_resnet50"
    
    fake_dir = None
    real_dir = None
    
    # Find fake adversarial images
    for ode_type in ["ode_custom_apgd-ce", "ode_euler_apgd-ce"]:
        base_dir = fake_base / ode_type / "seed42" / "data0" / "adv_images_rerun"
        if base_dir.exists():
            target_dir = base_dir / f"full_dire_resnet50_{data_gen}_test"
            if target_dir.exists():
                fake_dir = target_dir
                break
            for subdir in base_dir.iterdir():
                if subdir.is_dir() and "full_" in subdir.name:
                    fake_dir = subdir
                    break
            if fake_dir:
                break
    
    # Find real adversarial images
    for ode_type in ["ode_custom_apgd-ce", "ode_euler_apgd-ce"]:
        base_dir = real_base / ode_type / "seed42" / "data0" / "adv_images_rerun"
        if base_dir.exists():
            target_dir = base_dir / f"full_dire_resnet50_{attack_gen}_test"
            if target_dir.exists():
                real_dir = target_dir
                break
            for subdir in base_dir.iterdir():
                if subdir.is_dir() and "full_" in subdir.name:
                    real_dir = subdir
                    break
            if real_dir:
                break
    
    # Load valid indices
    fake_include_indices = load_dire_valid_indices(attack_gen, data_gen, is_fake=True) if fake_dir else None
    real_include_indices = load_dire_valid_indices(attack_gen, data_gen, is_fake=False) if real_dir else None
    
    return fake_dir, real_dir, fake_include_indices, real_include_indices


def get_lare_adv_paths(attack_gen: str, data_gen: str) -> Tuple[List[Path], List[Path]]:
    """Get LaRE² adversarial image paths.

    Returns (fake_adv_dirs, real_adv_dirs) as lists.
    - Default: only attack_success
    - USE_CLEAN_RIGHT_ONLY: attack_success + attack_fail
    - USE_ALL_SAMPLES: attack_success + attack_fail + clean_wrong
    """
    fake_base = LARE_ADV_ROOT / f"{attack_gen}_on_{data_gen}" / f"{attack_gen}_on_{data_gen}"
    real_base = LARE_ADV_ROOT / f"{attack_gen}" / f"{attack_gen}_on_{attack_gen}"

    fake_dirs = []
    real_dirs = []

    # Determine which subdirectories to include
    if USE_ALL_SAMPLES:
        subdirs = ["attack_success", "attack_fail", "clean_wrong"]
    elif USE_CLEAN_RIGHT_ONLY:
        subdirs = ["attack_success", "attack_fail"]
    else:
        subdirs = ["attack_success"]

    for subdir in subdirs:
        fake_dir = fake_base / subdir / "fake"
        if fake_dir.exists():
            fake_dirs.append(fake_dir)
        real_dir = real_base / subdir / "real"
        if real_dir.exists():
            real_dirs.append(real_dir)

    return fake_dirs, real_dirs


def get_ae_adv_paths(attack_gen: str, data_gen: str) -> Tuple[List[Path], List[Path]]:
    """Get AeroBlade adversarial image paths.

    Returns (fake_adv_dirs, real_adv_dirs) as lists.
    - Default: only attack_success
    - USE_CLEAN_RIGHT_ONLY: attack_success + attack_fail
    - USE_ALL_SAMPLES: attack_success + attack_fail + clean_wrong
    """
    fake_base = AE_ADV_ROOT / f"{attack_gen}_on_{data_gen}" / f"{attack_gen}_on_{data_gen}"
    real_base = AE_ADV_ROOT / f"{attack_gen}" / f"{attack_gen}_on_{attack_gen}"

    fake_dirs = []
    real_dirs = []

    # Determine which subdirectories to include
    if USE_ALL_SAMPLES:
        subdirs = ["attack_success", "attack_fail", "clean_wrong"]
    elif USE_CLEAN_RIGHT_ONLY:
        subdirs = ["attack_success", "attack_fail"]
    else:
        subdirs = ["attack_success"]

    for subdir in subdirs:
        fake_dir = fake_base / subdir / "fake"
        if fake_dir.exists():
            fake_dirs.append(fake_dir)
        real_dir = real_base / subdir / "real"
        if real_dir.exists():
            real_dirs.append(real_dir)

    return fake_dirs, real_dirs


# ==============================================================================
# Evaluation helpers for each detector
# ==============================================================================
def eval_on_dire(fake_dirs, real_dirs, 
                 detector: DIREDetector, eval_gen: str, args, desc_prefix: str = "",
                 fake_include_indices: Optional[set] = None, real_include_indices: Optional[set] = None) -> Tuple[float, float, float, int, int]:
    """Evaluate adversarial images on DIRE detector.
    
    Returns (robust_acc, fake_acc, real_acc, fake_total, real_total)
    
    Args:
        fake_dirs: List of directories containing adversarial fake images
        real_dirs: List of directories containing adversarial real images
        fake_include_indices: Set of valid sample indices to include for fake images
        real_include_indices: Set of valid sample indices to include for real images
    """
    fake_correct, fake_total = 0, 0
    real_correct, real_total = 0, 0
    
    detector.load_models(eval_gen)
    
    # Evaluate fake images from all directories
    for fake_dir in fake_dirs:
        if fake_dir is not None and fake_dir.exists():
            fake_dataset = AdvImageDataset(fake_dir, data_size=256, limit=args.limit, label=1, 
                                            include_indices=fake_include_indices)
            if len(fake_dataset) > 0:
                loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                for images, labels, _ in tqdm(loader, desc=f"{desc_prefix}DIRE fake", leave=False):
                    images = images.to(args.device)
                    preds, _ = detector.predict(images)
                    labels_t = torch.tensor(labels).to(args.device)
                    fake_correct += (preds == labels_t).sum().item()
                    fake_total += len(labels)
    
    # Evaluate real images from all directories
    for real_dir in real_dirs:
        if real_dir is not None and real_dir.exists():
            real_dataset = AdvImageDataset(real_dir, data_size=256, limit=args.limit, label=0,
                                            include_indices=real_include_indices)
            if len(real_dataset) > 0:
                loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                for images, labels, _ in tqdm(loader, desc=f"{desc_prefix}DIRE real", leave=False):
                    images = images.to(args.device)
                    preds, _ = detector.predict(images)
                    labels_t = torch.tensor(labels).to(args.device)
                    real_correct += (preds == labels_t).sum().item()
                    real_total += len(labels)
    
    # Compute robust accuracy (weighted by sample count)
    if fake_total > 0 and real_total > 0:
        fake_acc = fake_correct / fake_total * 100
        real_acc = real_correct / real_total * 100
        robust_acc = (fake_correct + real_correct) / (fake_total + real_total) * 100
        return robust_acc, fake_acc, real_acc, fake_total, real_total
    elif fake_total > 0:
        fake_acc = fake_correct / fake_total * 100
        return fake_acc, fake_acc, 0.0, fake_total, 0
    elif real_total > 0:
        real_acc = real_correct / real_total * 100
        return real_acc, 0.0, real_acc, 0, real_total
    return 0.0, 0.0, 0.0, 0, 0


def eval_on_lare(fake_dirs, real_dirs,
                 detector: LaREDetector, eval_gen: str, args, desc_prefix: str = "",
                 fake_include_indices: Optional[set] = None, real_include_indices: Optional[set] = None) -> Tuple[float, float, float, int, int]:
    """Evaluate adversarial images on LaRE² detector.
    
    Returns (robust_acc, fake_acc, real_acc, fake_total, real_total)
    
    Args:
        fake_dirs: List of directories containing adversarial fake images
        real_dirs: List of directories containing adversarial real images
        fake_include_indices: Set of valid sample indices to include for fake images (for DIRE attack)
        real_include_indices: Set of valid sample indices to include for real images (for DIRE attack)
    """
    fake_correct, fake_total = 0, 0
    real_correct, real_total = 0, 0
    
    detector.load_classifier(eval_gen)
    
    # Process all fake directories
    for fake_dir in fake_dirs:
        if fake_dir is not None and fake_dir.exists():
            fake_dataset = AdvImageDataset(fake_dir, data_size=224, limit=args.limit, label=1,
                                            include_indices=fake_include_indices)
            if len(fake_dataset) > 0:
                loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                for images, labels, _ in tqdm(loader, desc=f"{desc_prefix}LaRE fake", leave=False):
                    images = images.to(args.device)
                    preds, _ = detector.predict(images)
                    labels_t = torch.tensor(labels).to(args.device)
                    fake_correct += (preds == labels_t).sum().item()
                    fake_total += len(labels)
    
    # Process all real directories
    for real_dir in real_dirs:
        if real_dir is not None and real_dir.exists():
            real_dataset = AdvImageDataset(real_dir, data_size=224, limit=args.limit, label=0,
                                            include_indices=real_include_indices)
            if len(real_dataset) > 0:
                loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                for images, labels, _ in tqdm(loader, desc=f"{desc_prefix}LaRE real", leave=False):
                    images = images.to(args.device)
                    preds, _ = detector.predict(images)
                    labels_t = torch.tensor(labels).to(args.device)
                    real_correct += (preds == labels_t).sum().item()
                    real_total += len(labels)
    
    # Compute robust accuracy (weighted by sample count)
    if fake_total > 0 and real_total > 0:
        fake_acc = fake_correct / fake_total * 100
        real_acc = real_correct / real_total * 100
        robust_acc = (fake_correct + real_correct) / (fake_total + real_total) * 100
        return robust_acc, fake_acc, real_acc, fake_total, real_total
    elif fake_total > 0:
        fake_acc = fake_correct / fake_total * 100
        return fake_acc, fake_acc, 0.0, fake_total, 0
    elif real_total > 0:
        real_acc = real_correct / real_total * 100
        return real_acc, 0.0, real_acc, 0, real_total
    return 0.0, 0.0, 0.0, 0, 0


def eval_on_aeroblade(fake_dirs, real_dirs,
                      detector: AEDetector, threshold: float, sense: str, args, desc_prefix: str = "",
                      fake_include_indices: Optional[set] = None, real_include_indices: Optional[set] = None) -> Tuple[float, float, float, int, int]:
    """Evaluate adversarial images on AeroBlade detector.
    
    Returns (robust_acc, fake_acc, real_acc, fake_total, real_total)
    
    Args:
        fake_dirs: List of directories containing adversarial fake images
        real_dirs: List of directories containing adversarial real images
        threshold: Detection threshold
        sense: 'ge' means dist >= threshold -> fake, 'le' means dist <= threshold -> fake
        fake_include_indices: Set of valid sample indices to include for fake images (for DIRE attack)
        real_include_indices: Set of valid sample indices to include for real images (for DIRE attack)
    """
    fake_correct, fake_total = 0, 0
    real_correct, real_total = 0, 0
    
    detector.reset_generator()
    
    # Process all fake directories
    for fake_dir in fake_dirs:
        if fake_dir is not None and fake_dir.exists():
            fake_dataset = AdvImageDataset(fake_dir, data_size=512, limit=args.limit, label=1,
                                            include_indices=fake_include_indices)
            if len(fake_dataset) > 0:
                loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                for images, labels, _ in tqdm(loader, desc=f"{desc_prefix}AE fake", leave=False):
                    images = images.to(args.device)
                    distances = detector.compute_distances(images)
                    if sense == "ge":
                        preds = (distances >= threshold).long()
                    else:
                        preds = (distances <= threshold).long()
                    labels_t = torch.tensor(labels).to(args.device)
                    fake_correct += (preds == labels_t).sum().item()
                    fake_total += len(labels)
    
    # Process all real directories
    for real_dir in real_dirs:
        if real_dir is not None and real_dir.exists():
            real_dataset = AdvImageDataset(real_dir, data_size=512, limit=args.limit, label=0,
                                            include_indices=real_include_indices)
            if len(real_dataset) > 0:
                loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                for images, labels, _ in tqdm(loader, desc=f"{desc_prefix}AE real", leave=False):
                    images = images.to(args.device)
                    distances = detector.compute_distances(images)
                    if sense == "ge":
                        preds = (distances >= threshold).long()
                    else:
                        preds = (distances <= threshold).long()
                    labels_t = torch.tensor(labels).to(args.device)
                    real_correct += (preds == labels_t).sum().item()
                    real_total += len(labels)
    
    # Compute robust accuracy (weighted by sample count)
    if fake_total > 0 and real_total > 0:
        fake_acc = fake_correct / fake_total * 100
        real_acc = real_correct / real_total * 100
        robust_acc = (fake_correct + real_correct) / (fake_total + real_total) * 100
        return robust_acc, fake_acc, real_acc, fake_total, real_total
    elif fake_total > 0:
        fake_acc = fake_correct / fake_total * 100
        return fake_acc, fake_acc, 0.0, fake_total, 0
    elif real_total > 0:
        real_acc = real_correct / real_total * 100
        return real_acc, 0.0, real_acc, 0, real_total
    return 0.0, 0.0, 0.0, 0, 0


# ==============================================================================
# Main evaluation: Cross-Both Transferability
# ==============================================================================
def eval_cross_both(attack_method: str, eval_method: str, args,
                    dire_detector: DIREDetector,
                    lare_detector: LaREDetector,
                    ae_detector: AEDetector,
                    ae_thresholds: Dict[str, Tuple[float, str]]) -> np.ndarray:
    """
    Evaluate cross-both transferability: attack_method -> eval_method
    
    Matrix[i, j] = adversarial examples from attack_method (cls_i attacking data_j)
                   evaluated on eval_method cls_j.

    - Row i:    attack classifier training dataset (ADM, FLUX, SD1.5, VQDM)
    - Column j: eval classifier training dataset = attacked dataset (ADM, FLUX, SD1.5, VQDM)
    
    Args:
        ae_thresholds: Dict mapping gen -> (threshold, sense)
    
    Note: For LaRE and AeroBlade attack methods, samples that were initially misclassified
          (stored in clean_wrong directories) are excluded from evaluation.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Cross-Both: {METHOD_DISPLAY[attack_method]} → {METHOD_DISPLAY[eval_method]}", flush=True)
    print(f"{'='*60}", flush=True)
    
    mat_acc = np.zeros((len(GENS), len(GENS)), dtype=np.float32)
    
    for i, attack_gen in enumerate(GENS):
        for j, data_gen in enumerate(GENS):
            # attack_gen: classifier training dataset
            # data_gen: attacked dataset = eval classifier dataset
            
            # Get adversarial images from attack_method
            # Get adversarial paths
            # DIRE returns (fake_dir, real_dir, fake_include_indices, real_include_indices)
            # LaRE/AE return (fake_dirs, real_dirs) as lists
            fake_include_indices, real_include_indices = None, None
            
            if attack_method == "dire":
                fake_dir, real_dir, fake_include_indices, real_include_indices = get_dire_adv_paths(attack_gen, data_gen)
                # Wrap single paths in lists for consistency
                fake_dirs = [fake_dir] if fake_dir is not None else []
                real_dirs = [real_dir] if real_dir is not None else []
            elif attack_method == "lare":
                fake_dirs, real_dirs = get_lare_adv_paths(attack_gen, data_gen)
            else:  # aeroblade
                fake_dirs, real_dirs = get_ae_adv_paths(attack_gen, data_gen)
            
            if len(fake_dirs) == 0 and len(real_dirs) == 0:
                print(f"  [{GEN_DISPLAY[attack_gen]}→{GEN_DISPLAY[data_gen]}] No adv images", flush=True)
                continue
            
            # Evaluate on eval_method with cls_{data_gen}
            desc_prefix = f"{GEN_DISPLAY[attack_gen]}->{GEN_DISPLAY[data_gen]} "
            
            if eval_method == "dire":
                acc, fake_acc, real_acc, fake_n, real_n = eval_on_dire(
                    fake_dirs, real_dirs, 
                    dire_detector, data_gen, args, desc_prefix,
                    fake_include_indices, real_include_indices)
            elif eval_method == "lare":
                acc, fake_acc, real_acc, fake_n, real_n = eval_on_lare(
                    fake_dirs, real_dirs, lare_detector, data_gen, args, desc_prefix,
                    fake_include_indices, real_include_indices)
            else:  # aeroblade
                ae_thr, ae_sense = ae_thresholds.get(data_gen, (-0.024, "ge"))
                acc, fake_acc, real_acc, fake_n, real_n = eval_on_aeroblade(
                    fake_dirs, real_dirs, ae_detector, ae_thr, ae_sense, args, desc_prefix,
                    fake_include_indices, real_include_indices)
            
            mat_acc[i, j] = acc
            
            print(f"  [{GEN_DISPLAY[attack_gen]}→{GEN_DISPLAY[data_gen]}] "
                  f"Robust={acc:.2f}% | Fake={fake_acc:.2f}%({fake_n}) Real={real_acc:.2f}%({real_n})", flush=True)
    
    return mat_acc


def main():
    global USE_CLEAN_RIGHT_ONLY, USE_ALL_SAMPLES
    global DIRE_ADV_ROOT, LARE_ADV_ROOT, AE_ADV_ROOT, DIRE_VALID_INDICES_DIR, DIRE_CLEAN_RIGHT_ONLY_DIR

    parser = argparse.ArgumentParser(description="Cross-Both Adversarial Transferability Evaluation")
    parser.add_argument("--output_dir", type=str, default="./figures/cross_both_transfer")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per attack pair")
    parser.add_argument("--aeroblade_csv", type=str,
                        default=os.environ.get("AE_CALIB_CSV", "/path/to/distances_calib.csv"),
                        help="Path to AeroBlade calibration distances CSV")
    parser.add_argument("--dire_adv_root", type=str,
                        default=os.environ.get("DIRE_ADV_ROOT", "/path/to/dire_adversarial_images"))
    parser.add_argument("--lare_adv_root", type=str,
                        default=os.environ.get("LARE_ADV_ROOT", "/path/to/lare_adversarial_images"))
    parser.add_argument("--ae_adv_root", type=str,
                        default=os.environ.get("AE_ADV_ROOT", "/path/to/aeroblade_adversarial_images"))
    parser.add_argument("--dire_valid_indices_dir", type=str,
                        default=os.environ.get("DIRE_VALID_INDICES_DIR", ""))
    parser.add_argument("--dire_clean_right_only_dir", type=str,
                        default=os.environ.get("DIRE_CLEAN_RIGHT_ONLY_DIR", ""))
    parser.add_argument("--clean_right_only", action="store_true",
                        help="Use clean_right samples instead of attack_success")
    parser.add_argument("--all_samples", action="store_true", help="Evaluate all samples without restrictions.")
    args = parser.parse_args()

    DIRE_ADV_ROOT = Path(args.dire_adv_root)
    LARE_ADV_ROOT = Path(args.lare_adv_root)
    AE_ADV_ROOT = Path(args.ae_adv_root)
    DIRE_VALID_INDICES_DIR = Path(args.dire_valid_indices_dir) if args.dire_valid_indices_dir else None
    DIRE_CLEAN_RIGHT_ONLY_DIR = Path(args.dire_clean_right_only_dir) if args.dire_clean_right_only_dir else None

    USE_CLEAN_RIGHT_ONLY = args.clean_right_only
    USE_ALL_SAMPLES = args.all_samples
    if USE_ALL_SAMPLES:
        print(f"[INFO] Using all_samples mode: no filtering applied")
        print(f"[INFO] DIRE: all images loaded; LaRE/AeroBlade: both attack_success and attack_fail directories")
    elif USE_CLEAN_RIGHT_ONLY:
        print(f"[INFO] Using clean_right_only mode: indices from {DIRE_CLEAN_RIGHT_ONLY_DIR}")
        print(f"[INFO] LaRE/AeroBlade will include both attack_success and attack_fail directories")
    
    set_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Initialize detectors
    # ──────────────────────────────────────────────────────────────────
    print("Initializing detectors...")
    
    dire_detector = DIREDetector(device=args.device, t=20)
    
    lare_detector = LaREDetector(device=args.device, t=200, ensemble_size=4)
    lare_detector.load_sd()
    
    ae_detector = AEDetector(device=args.device, seed=42)
    ae_detector.load_models()
    
    # Load AeroBlade thresholds for each generator
    import pandas as pd
    calib_csv_path = Path(args.aeroblade_csv)
    ae_thresholds: Dict[str, Tuple[float, str]] = {}
    
    if calib_csv_path.exists():
        print(f"Loading calibration CSV: {calib_csv_path}", flush=True)
        calib_df = pd.read_csv(calib_csv_path)
        
        # Compute optimal threshold for each generator
        for gen in GENS:
            dists, labels = AEDetector.extract_distances_from_df(calib_df, gen)
            if len(dists) > 0:
                threshold, sense = AEDetector._find_optimal_threshold(dists, labels)
                acc = ((dists >= threshold if sense == "ge" else dists <= threshold).astype(int) == labels).mean() * 100
                ae_thresholds[gen] = (threshold, sense)
                print(f"  [thr_csv] {gen}: acc={acc:.2f}% thr={threshold:.6f} sense={sense}", flush=True)
            else:
                ae_thresholds[gen] = (-0.024, "ge")
                print(f"  [thr_csv] {gen}: no data, using default", flush=True)
    else:
        print(f"[warning] Calibration CSV not found: {calib_csv_path}", flush=True)
        for gen in GENS:
            ae_thresholds[gen] = (-0.024, "ge")

    # ──────────────────────────────────────────────────────────────────
    # Cross-both pairs (6 combinations, excluding self->self)
    # ──────────────────────────────────────────────────────────────────
    cross_pairs = [
        ("dire", "lare"),
        ("dire", "aeroblade"),
        ("lare", "dire"),
        ("lare", "aeroblade"),
        ("aeroblade", "dire"),
        ("aeroblade", "lare"),
    ]
    
    all_results = {}
    
    for attack_method, eval_method in cross_pairs:
        mat_acc = eval_cross_both(
            attack_method, eval_method, args,
            dire_detector, lare_detector, ae_detector, ae_thresholds
        )
        
        key = f"{attack_method}_to_{eval_method}"
        all_results[key] = mat_acc
        
        # Draw heatmap
        if mat_acc.any():
            # Generate title with proper LaTeX
            attack_disp = METHOD_DISPLAY[attack_method]
            eval_disp = METHOD_DISPLAY[eval_method]
            title = f'{attack_disp} → {eval_disp}'
            
            draw_single_heatmap(
                mat_acc,
                title,
                vmin=0, vmax=100,
                cmap='Blues',
                filename_base=f'heatmap_crossboth_{attack_method}_to_{eval_method}',
                out_dir=output_dir,
                xlabel=f'{eval_disp} Classifier',
                ylabel=f'{attack_disp} Classifier'
            )
            np.save(output_dir / f"{key}_acc_matrix.npy", mat_acc)
            print(f"\n{title} Matrix:\n{mat_acc}", flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary - Cross-Both Transferability")
    print(f"{'='*60}")
    
    for key, mat in all_results.items():
        attack_method, eval_method = key.replace("_to_", " -> ").split(" -> ")
        print(f"\n{METHOD_DISPLAY.get(attack_method, attack_method)} → "
              f"{METHOD_DISPLAY.get(eval_method, eval_method)}:")
        print(f"  Mean ACC: {mat.mean():.2f}%")
        print(f"  Diagonal (same dataset): {np.diag(mat).mean():.2f}%")
        off_diag = mat[~np.eye(mat.shape[0], dtype=bool)]
        print(f"  Off-diagonal (cross dataset): {off_diag.mean():.2f}%")

    print(f"\n[done] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
