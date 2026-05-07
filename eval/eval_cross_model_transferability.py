#!/usr/bin/env python3
"""
Cross-Model Adversarial Transferability Evaluation.

Evaluates adversarial transferability across generator-specific classifiers.
Heatmap (i, j): classifier trained on generator i evaluated on adversarial
examples crafted against classifier j.

Data directory structure:
- DIRE:       <dire_adv_root>/fake_cls_{attack}_on_{data}/dire_resnet50/...
              <dire_adv_root>/real_cls_{attack}/...
- LaRE:       <lare_adv_root>/{attack}_on_{data}/{attack}_on_{data}/attack_success/fake/
- AeroBlade:  <ae_adv_root>/{attack}_on_{data}/{attack}_on_{data}/attack_success/fake/

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
DISPLAY_NAMES = {"adm": "ADM", "flux": "FLUX", "sdv5": "SD1.5", "vqdm": "VQDM"}

# Paths are configured via argparse arguments (see main())
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
# ECCV-style Heatmap (consistent with plot_heatmap_transferability.py)
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
                        xlabel='Evaluation Classifier', ylabel='Attack Classifier'):
    """Draw a single annotated heatmap as separate figure."""
    if row_labels is None:
        row_labels = [DISPLAY_NAMES.get(g, g) for g in GENS]
    if col_labels is None:
        col_labels = [DISPLAY_NAMES.get(g, g) for g in GENS]
    
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
            # Choose text color for readability (white on dark, black on light)
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
            include_indices: Set of valid sample indices to include (for DIRE only)
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
                print(f"  [warning] Dir not found: {img_dir}")
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
                match = re.search(r'sample_(\d+)_adv', s.stem)
                if match:
                    idx = int(match.group(1))
                    if idx in include_indices:
                        filtered_samples.append(s)
            self.samples = filtered_samples
            print(f"  [include] {original_count} -> {len(self.samples)} valid samples (clean_right AND attack_success)")
        
        if limit and limit > 0:
            self.samples = self.samples[:limit]
        
        print(f"  [dataset] Found {len(self.samples)} images in {img_dir.name}")

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
# Find adversarial image directories
# ==============================================================================
def get_dire_adv_paths(attack_gen: str, data_gen: str) -> Tuple[Optional[Path], Optional[Path], Optional[set], Optional[set]]:
    """Get DIRE adversarial image paths for attack_gen classifier attacking data_gen dataset.
    
    Returns (fake_adv_dir, real_adv_dir, fake_exclude_indices, real_exclude_indices).
    
    Actual path structure:
    - fake: exp_attacks.../fake_cls_{attack}_on_{data}/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_{data}_test/
    - real: exp_attacks.../real_cls_{attack}/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_{attack}_test/
    """
    fake_base = DIRE_ADV_ROOT / f"fake_cls_{attack_gen}_on_{data_gen}" / "dire_resnet50"
    real_base = DIRE_ADV_ROOT / f"real_cls_{attack_gen}" / "dire_resnet50"
    
    fake_dir = None
    real_dir = None
    
    # Find fake adversarial images
    for ode_type in ["ode_custom_apgd-ce", "ode_euler_apgd-ce"]:
        base_dir = fake_base / ode_type / "seed42" / "data0" / "adv_images_rerun"
        if base_dir.exists():
            # Look for full_dire_resnet50_{data_gen}_test directory
            target_dir = base_dir / f"full_dire_resnet50_{data_gen}_test"
            if target_dir.exists():
                fake_dir = target_dir
                break
            # Fallback: check any matching directory
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
            # Real images attacked with cls_{attack_gen}, so look for full_dire_resnet50_{attack_gen}_test
            target_dir = base_dir / f"full_dire_resnet50_{attack_gen}_test"
            if target_dir.exists():
                real_dir = target_dir
                break
            # Fallback
            for subdir in base_dir.iterdir():
                if subdir.is_dir() and "full_" in subdir.name:
                    real_dir = subdir
                    break
            if real_dir:
                break
    
    # Load valid indices (clean_right AND attack_success)
    fake_include_indices = load_dire_valid_indices(attack_gen, data_gen, is_fake=True) if fake_dir else None
    real_include_indices = load_dire_valid_indices(attack_gen, data_gen, is_fake=False) if real_dir else None
    
    return fake_dir, real_dir, fake_include_indices, real_include_indices


def get_lare_adv_paths(attack_gen: str, data_gen: str) -> Tuple[List[Path], List[Path]]:
    """Get LaRE adversarial image paths for attack_gen classifier attacking data_gen dataset.

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
# Evaluation Functions
# ==============================================================================
def eval_dire_cross_model(args) -> np.ndarray:
    """
    Evaluate DIRE cross-model transferability.
    
    Matrix[i, j] = attack on cls_i with data_j -> test on cls_j

    Robust Accuracy = (real_acc + fake_acc) / 2
    - real_acc: detection accuracy on real adversarial samples (should predict 0)
    - fake_acc: detection accuracy on fake adversarial samples (should predict 1)
    
    Note: Samples that were initially misclassified (identified in attack logs as "correct: False")
          are included from evaluation using pre-extracted index files.
    """
    print("\n" + "="*60)
    print("DIRE Cross-Model Transferability Evaluation")
    print("="*60)

    detector = DIREDetector(device=args.device, t=20)
    mat_acc = np.zeros((len(GENS), len(GENS)), dtype=np.float32)

    for i, attack_gen in enumerate(GENS):
        for j, data_gen in enumerate(GENS):
            print(f"\n[DIRE] Attack: cls_{attack_gen} on {data_gen} -> Eval: cls_{data_gen}")
            
            # Load the evaluation classifier (cls_data_gen)
            detector.load_models(data_gen)
            
            # Get adversarial image paths and include indices
            fake_dir, real_dir, fake_include_indices, real_include_indices = get_dire_adv_paths(attack_gen, data_gen)
            
            # Log inclusion info
            if fake_include_indices:
                print(f"  Including {len(fake_include_indices)} valid fake samples (clean_right AND attack_success)")
            if real_include_indices:
                print(f"  Including {len(real_include_indices)} valid real samples (clean_right AND attack_success)")
            
            fake_correct, fake_total = 0, 0
            real_correct, real_total = 0, 0
            
            # ───────────────────────────────────────────────────────────
            # Evaluate FAKE adversarial images (label=1, should predict fake)
            # ───────────────────────────────────────────────────────────
            if fake_dir is not None and fake_dir.exists():
                fake_dataset = AdvImageDataset(fake_dir, data_size=256, limit=args.limit, label=1,
                                                include_indices=fake_include_indices)
                if len(fake_dataset) > 0:
                    loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                    for images, labels, _ in tqdm(loader, desc=f"DIRE fake {attack_gen}->{data_gen}", leave=False):
                        images = images.to(args.device)
                        preds, _ = detector.predict(images)
                        labels_t = torch.tensor(labels).to(args.device)
                        fake_correct += (preds == labels_t).sum().item()
                        fake_total += len(labels)
                    print(f"  Fake: {fake_dir.name}")
                else:
                    print(f"  [skip] No fake samples in {fake_dir}")
            else:
                print(f"  [skip] Fake dir not found")
            
            # ───────────────────────────────────────────────────────────
            # Evaluate REAL adversarial images (label=0, should predict real)
            # ───────────────────────────────────────────────────────────
            if real_dir is not None and real_dir.exists():
                real_dataset = AdvImageDataset(real_dir, data_size=256, limit=args.limit, label=0,
                                                include_indices=real_include_indices)
                if len(real_dataset) > 0:
                    loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                    for images, labels, _ in tqdm(loader, desc=f"DIRE real {attack_gen}->{data_gen}", leave=False):
                        images = images.to(args.device)
                        preds, _ = detector.predict(images)
                        labels_t = torch.tensor(labels).to(args.device)
                        real_correct += (preds == labels_t).sum().item()
                        real_total += len(labels)
                    print(f"  Real: {real_dir.name}")
                else:
                    print(f"  [skip] No real samples in {real_dir}")
            else:
                print(f"  [skip] Real dir not found")
            
            # ───────────────────────────────────────────────────────────
            # Compute combined robust accuracy (weighted by sample count)
            # ───────────────────────────────────────────────────────────
            if fake_total > 0 and real_total > 0:
                fake_acc = fake_correct / fake_total * 100
                real_acc = real_correct / real_total * 100
                # Weighted average by sample count
                robust_acc = (fake_correct + real_correct) / (fake_total + real_total) * 100
                mat_acc[i, j] = robust_acc
                print(f"  Fake ACC: {fake_acc:.2f}% ({fake_correct}/{fake_total})")
                print(f"  Real ACC: {real_acc:.2f}% ({real_correct}/{real_total})")
                print(f"  Robust ACC: {robust_acc:.2f}% (weighted, total={fake_total+real_total})")
            elif fake_total > 0:
                fake_acc = fake_correct / fake_total * 100
                mat_acc[i, j] = fake_acc
                print(f"  Fake-only ACC: {fake_acc:.2f}% ({fake_correct}/{fake_total})")
            elif real_total > 0:
                real_acc = real_correct / real_total * 100
                mat_acc[i, j] = real_acc
                print(f"  Real-only ACC: {real_acc:.2f}% ({real_correct}/{real_total})")
            else:
                print(f"  [skip] No samples evaluated")

    return mat_acc

    return mat_acc


def eval_lare_cross_model(args) -> np.ndarray:
    """
    Evaluate LaRE cross-model transferability.

    Robust Accuracy = (real_acc + fake_acc) / 2

    Note: attack_success directory already contains only clean_right AND attack_success samples.
    """
    print("\n" + "="*60)
    print("LaRE Cross-Model Transferability Evaluation")
    print("="*60)

    detector = LaREDetector(device=args.device, t=200, ensemble_size=4)
    detector.load_sd()
    mat_acc = np.zeros((len(GENS), len(GENS)), dtype=np.float32)

    for i, attack_gen in enumerate(GENS):
        for j, data_gen in enumerate(GENS):
            print(f"\n[LaRE] Attack: cls_{attack_gen} on {data_gen} -> Eval: cls_{data_gen}")
            
            # Load the evaluation classifier
            detector.load_classifier(data_gen)
            
            # Get adversarial image paths (returns lists)
            fake_dirs, real_dirs = get_lare_adv_paths(attack_gen, data_gen)
            
            fake_correct, fake_total = 0, 0
            real_correct, real_total = 0, 0
            
            # ───────────────────────────────────────────────────────────
            # Evaluate FAKE adversarial images (label=1)
            # ───────────────────────────────────────────────────────────
            if fake_dirs:
                fake_dataset = AdvImageDataset(fake_dirs, data_size=224, limit=args.limit, label=1)
                if len(fake_dataset) > 0:
                    loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                    for images, labels, _ in tqdm(loader, desc=f"LaRE fake {attack_gen}->{data_gen}", leave=False):
                        images = images.to(args.device)
                        preds, _ = detector.predict(images)
                        labels_t = torch.tensor(labels).to(args.device)
                        fake_correct += (preds == labels_t).sum().item()
                        fake_total += len(labels)
                else:
                    print(f"  [skip] No fake samples")
            else:
                print(f"  [skip] Fake dirs not found")
            
            # ───────────────────────────────────────────────────────────
            # Evaluate REAL adversarial images (label=0)
            # ───────────────────────────────────────────────────────────
            if real_dirs:
                real_dataset = AdvImageDataset(real_dirs, data_size=224, limit=args.limit, label=0)
                if len(real_dataset) > 0:
                    loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                    for images, labels, _ in tqdm(loader, desc=f"LaRE real {attack_gen}->{data_gen}", leave=False):
                        images = images.to(args.device)
                        preds, _ = detector.predict(images)
                        labels_t = torch.tensor(labels).to(args.device)
                        real_correct += (preds == labels_t).sum().item()
                        real_total += len(labels)
                else:
                    print(f"  [skip] No real samples")
            else:
                print(f"  [skip] Real dirs not found")
            
            # ───────────────────────────────────────────────────────────
            # Compute combined robust accuracy (weighted by sample count)
            # ───────────────────────────────────────────────────────────
            if fake_total > 0 and real_total > 0:
                fake_acc = fake_correct / fake_total * 100
                real_acc = real_correct / real_total * 100
                # Weighted average by sample count
                robust_acc = (fake_correct + real_correct) / (fake_total + real_total) * 100
                mat_acc[i, j] = robust_acc
                print(f"  Fake ACC: {fake_acc:.2f}% ({fake_correct}/{fake_total})")
                print(f"  Real ACC: {real_acc:.2f}% ({real_correct}/{real_total})")
                print(f"  Robust ACC: {robust_acc:.2f}% (weighted, total={fake_total+real_total})")
            elif fake_total > 0:
                fake_acc = fake_correct / fake_total * 100
                mat_acc[i, j] = fake_acc
                print(f"  Fake-only ACC: {fake_acc:.2f}% ({fake_correct}/{fake_total})")
            elif real_total > 0:
                real_acc = real_correct / real_total * 100
                mat_acc[i, j] = real_acc
                print(f"  Real-only ACC: {real_acc:.2f}% ({real_correct}/{real_total})")
            else:
                print(f"  [skip] No samples evaluated")

    return mat_acc


def eval_ae_cross_model(args) -> np.ndarray:
    """
    Evaluate AeroBlade cross-model transferability.
    
    Note: AeroBlade uses threshold-based detection; calibration data is required.

    Robust Accuracy = (real_acc + fake_acc) / 2
    Each data_gen uses its own calibrated threshold and sense direction.

    Note: attack_success directory already contains only clean_right AND attack_success samples.
    """
    print("\n" + "="*60, flush=True)
    print("AEROBLADE Cross-Model Transferability Evaluation", flush=True)
    print("="*60, flush=True)

    detector = AEDetector(device=args.device, seed=42)
    detector.load_models()
    
    mat_acc = np.zeros((len(GENS), len(GENS)), dtype=np.float32)
    
    # Load calibration thresholds for each generator
    calib_csv_path = Path(args.aeroblade_csv)
    ae_thresholds: Dict[str, Tuple[float, str]] = {}
    
    import pandas as pd
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

    for i, attack_gen in enumerate(GENS):
        for j, data_gen in enumerate(GENS):
            # Use threshold for data_gen (the evaluation classifier's dataset)
            threshold, sense = ae_thresholds.get(data_gen, (-0.024, "ge"))
            print(f"\n[AE] Attack: cls_{attack_gen} on {data_gen} -> Eval with thr={threshold:.6f}, sense={sense}", flush=True)
            
            # Get adversarial image paths (returns lists)
            fake_dirs, real_dirs = get_ae_adv_paths(attack_gen, data_gen)
            
            fake_correct, fake_total = 0, 0
            real_correct, real_total = 0, 0
            
            # Reset generator for reproducibility
            detector.reset_generator()
            
            # ───────────────────────────────────────────────────────────
            # Evaluate FAKE adversarial images (label=1)
            # ───────────────────────────────────────────────────────────
            if fake_dirs:
                fake_dataset = AdvImageDataset(fake_dirs, data_size=512, limit=args.limit, label=1)
                if len(fake_dataset) > 0:
                    loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                    for images, labels, _ in tqdm(loader, desc=f"AE fake {attack_gen}->{data_gen}", leave=False):
                        images = images.to(args.device)
                        distances = detector.compute_distances(images)
                        if sense == "ge":
                            preds = (distances >= threshold).long()
                        else:
                            preds = (distances <= threshold).long()
                        labels_t = torch.tensor(labels).to(args.device)
                        fake_correct += (preds == labels_t).sum().item()
                        fake_total += len(labels)
                else:
                    print(f"  [skip] No fake samples")
            else:
                print(f"  [skip] Fake dirs not found")
            
            # ───────────────────────────────────────────────────────────
            # Evaluate REAL adversarial images (label=0)
            # ───────────────────────────────────────────────────────────
            if real_dirs:
                real_dataset = AdvImageDataset(real_dirs, data_size=512, limit=args.limit, label=0)
                if len(real_dataset) > 0:
                    loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                    for images, labels, _ in tqdm(loader, desc=f"AE real {attack_gen}->{data_gen}", leave=False):
                        images = images.to(args.device)
                        distances = detector.compute_distances(images)
                        if sense == "ge":
                            preds = (distances >= threshold).long()
                        else:
                            preds = (distances <= threshold).long()
                        labels_t = torch.tensor(labels).to(args.device)
                        real_correct += (preds == labels_t).sum().item()
                        real_total += len(labels)
                else:
                    print(f"  [skip] No real samples")
            else:
                print(f"  [skip] Real dirs not found")
            
            # ───────────────────────────────────────────────────────────
            # Compute combined robust accuracy (weighted by sample count)
            # ───────────────────────────────────────────────────────────
            if fake_total > 0 and real_total > 0:
                fake_acc = fake_correct / fake_total * 100
                real_acc = real_correct / real_total * 100
                # Weighted average by sample count
                robust_acc = (fake_correct + real_correct) / (fake_total + real_total) * 100
                mat_acc[i, j] = robust_acc
                print(f"  Fake ACC: {fake_acc:.2f}% ({fake_correct}/{fake_total})")
                print(f"  Real ACC: {real_acc:.2f}% ({real_correct}/{real_total})")
                print(f"  Robust ACC: {robust_acc:.2f}% (weighted, total={fake_total+real_total})")
            elif fake_total > 0:
                fake_acc = fake_correct / fake_total * 100
                mat_acc[i, j] = fake_acc
                print(f"  Fake-only ACC: {fake_acc:.2f}% ({fake_correct}/{fake_total})")
            elif real_total > 0:
                real_acc = real_correct / real_total * 100
                mat_acc[i, j] = real_acc
                print(f"  Real-only ACC: {real_acc:.2f}% ({real_correct}/{real_total})")
            else:
                print(f"  [skip] No samples evaluated")

    return mat_acc


def main():
    global USE_CLEAN_RIGHT_ONLY, USE_ALL_SAMPLES
    global DIRE_ADV_ROOT, LARE_ADV_ROOT, AE_ADV_ROOT, DIRE_VALID_INDICES_DIR, DIRE_CLEAN_RIGHT_ONLY_DIR

    parser = argparse.ArgumentParser(description="Cross-Model Adversarial Transferability Evaluation")
    parser.add_argument("--output_dir", type=str, default="./figures/cross_model_transfer")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per attack pair")
    parser.add_argument("--eval", type=str, default="all",
                        choices=["all", "dire", "lare", "aeroblade"],
                        help="Which detector to evaluate")
    parser.add_argument("--aeroblade_csv", type=str,
                        default=os.environ.get("AE_CALIB_CSV", "/path/to/distances_calib.csv"),
                        help="Path to AeroBlade calibration distances CSV")
    parser.add_argument("--dire_adv_root", type=str,
                        default=os.environ.get("DIRE_ADV_ROOT", "/path/to/dire_adversarial_images"),
                        help="Root of DIRE adversarial image outputs")
    parser.add_argument("--lare_adv_root", type=str,
                        default=os.environ.get("LARE_ADV_ROOT", "/path/to/lare_adversarial_images"),
                        help="Root of LaRE adversarial image outputs")
    parser.add_argument("--ae_adv_root", type=str,
                        default=os.environ.get("AE_ADV_ROOT", "/path/to/aeroblade_adversarial_images"),
                        help="Root of AeroBlade adversarial image outputs")
    parser.add_argument("--dire_valid_indices_dir", type=str,
                        default=os.environ.get("DIRE_VALID_INDICES_DIR", ""),
                        help="Directory containing DIRE valid index files (clean_right AND attack_success)")
    parser.add_argument("--dire_clean_right_only_dir", type=str,
                        default=os.environ.get("DIRE_CLEAN_RIGHT_ONLY_DIR", ""),
                        help="Directory containing DIRE clean_right only index files")
    parser.add_argument("--clean_right_only", action="store_true",
                        help="Use clean_right only indices (not limited to attack_success)")
    parser.add_argument("--all_samples", action="store_true", help="Evaluate all samples without restrictions.")
    args = parser.parse_args()

    DIRE_ADV_ROOT = Path(args.dire_adv_root)
    LARE_ADV_ROOT = Path(args.lare_adv_root)
    AE_ADV_ROOT = Path(args.ae_adv_root)
    DIRE_VALID_INDICES_DIR = Path(args.dire_valid_indices_dir) if args.dire_valid_indices_dir else None
    DIRE_CLEAN_RIGHT_ONLY_DIR = Path(args.dire_clean_right_only_dir) if args.dire_clean_right_only_dir else None
    
    # Set global flags for index selection
    USE_CLEAN_RIGHT_ONLY = args.clean_right_only
    USE_ALL_SAMPLES = args.all_samples
    if USE_ALL_SAMPLES:
        print(">>> Using all_samples mode: no filtering applied")
        print(">>> DIRE: all images loaded; LaRE/AeroBlade: both attack_success and attack_fail directories")
    elif USE_CLEAN_RIGHT_ONLY:
        print(">>> Using clean_right ONLY indices (not limited to attack_success)")
    else:
        print(">>> Using clean_right AND attack_success indices")

    set_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # DIRE
    # ──────────────────────────────────────────────────────────────────
    if args.eval in ["all", "dire"]:
        dire_acc = eval_dire_cross_model(args)
        if dire_acc.any():
            draw_single_heatmap(
                dire_acc,
                'DIRE Cross-Generator Transfer',
                vmin=0, vmax=100,
                cmap='Blues',
                filename_base='heatmap_crossgenerator_DIRE',
                out_dir=output_dir,
                xlabel='Evaluation Classifier',
                ylabel='Attack Classifier'
            )
            np.save(output_dir / "dire_crossgenerator_acc_matrix.npy", dire_acc)
            print(f"\nDIRE Matrix (before transpose):\n{dire_acc}")

    # ──────────────────────────────────────────────────────────────────
    # LaRE²
    # ──────────────────────────────────────────────────────────────────
    if args.eval in ["all", "lare"]:
        lare_acc = eval_lare_cross_model(args)
        if lare_acc.any():
            draw_single_heatmap(
                lare_acc,
                r'LaRE$^2$ Cross-Generator Transfer',
                vmin=0, vmax=100,
                cmap='Blues',
                filename_base='heatmap_crossgenerator_LaRE2',
                out_dir=output_dir,
                xlabel='Evaluation Classifier',
                ylabel='Attack Classifier'
            )
            np.save(output_dir / "lare_crossgenerator_acc_matrix.npy", lare_acc)
            print(f"\nLaRE² Matrix (before transpose):\n{lare_acc}")

    # ──────────────────────────────────────────────────────────────────
    # AeroBlade
    # ──────────────────────────────────────────────────────────────────
    if args.eval in ["all", "aeroblade"]:
        ae_acc = eval_ae_cross_model(args)
        if ae_acc.any():
            draw_single_heatmap(
                ae_acc,
                'AEROBLADE Cross-Generator Transfer',
                vmin=0, vmax=100,
                cmap='Blues',
                filename_base='heatmap_crossgenerator_AEROBLADE',
                out_dir=output_dir,
                xlabel='Evaluation Classifier',
                ylabel='Attack Classifier'
            )
            np.save(output_dir / "ae_crossgenerator_acc_matrix.npy", ae_acc)
            print(f"\nAeroBlade Matrix (before transpose):\n{ae_acc}")

    print(f"\n[done] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
