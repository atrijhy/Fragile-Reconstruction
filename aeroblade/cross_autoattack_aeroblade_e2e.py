#!/usr/bin/env python3
"""
Cross AutoAttack for AEROBLADE (true E2E, recomputes reconstruction error).

For each train_gen -> test_gen pair:
  1. Compute real/val and train_gen/val distances to calibrate threshold.
  2. Evaluate clean accuracy on test set.
  3. Run AutoAttack (apgd-ce) on real/test and test_gen/test images,
     recompute distances on perturbed images, evaluate attacked accuracy.
  4. Save adversarial images to <save_dir>/<train>_on_<test>/<category>/<real|fake>/.
  5. Generate clean/adv accuracy heatmaps.

Note: threshold is calibrated on val set; attack runs on test set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from autoattack import AutoAttack
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from aeroblade.distances import _PatchedLPIPS  # noqa: E402
from aeroblade.data import ImageFolder  # noqa: E402
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents  # noqa: E402

try:
    from diffusers import AutoencoderKL
    from diffusers.models import VQModel
except ImportError as e:
    raise ImportError("diffusers not installed") from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AeroBladeDetector(nn.Module):
    """
    Differentiable AEROBLADE detector with threshold offset.

    signed = (dist - thr) * dir, where dir=+1 means dist>=thr predicts fake.
    logits = [-signed*scale, signed*scale]
    """

    def __init__(
        self,
        repo_ids: List[str],
        distance_metric: str,
        device: torch.device,
        seed: int = 1,
        scale: float = 10.0,
        dist_sign: float = 1.0,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.distance_metric = distance_metric
        self.device = device
        self.seed = seed
        self.scale = scale
        self.dist_sign = dist_sign
        if distance_metric.startswith("lpips"):
            parts = distance_metric.split("_")
            self.lpips_net = parts[1]
            self.lpips_layer = int(parts[2])
        else:
            self.lpips_net = "vgg"
            self.lpips_layer = -1

        self.lpips_model = _PatchedLPIPS(net=self.lpips_net, spatial=True).to(device)
        self.lpips_model.eval()
        for p in self.lpips_model.parameters():
            p.requires_grad = False

        self.autoencoders = {}
        self._load_autoencoders()
        self.generator = torch.Generator(device=device).manual_seed(seed)

        self.thr = 0.0
        self.dir = 1.0

    def _load_autoencoders(self):
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        for repo_id in self.repo_ids:
            if "kandinsky-2" in repo_id:
                ae = VQModel.from_pretrained(repo_id, subfolder="movq", torch_dtype=dtype)
            else:
                ae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=dtype)
            ae = ae.to(self.device).eval()
            for p in ae.parameters():
                p.requires_grad = False
            self.autoencoders[repo_id] = ae

    def set_threshold(self, thr: float, sense: str):
        self.thr = float(thr)
        self.dir = 1.0 if sense == "ge" else -1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        b = x.shape[0]
        all_dist = []
        for repo_id, ae in self.autoencoders.items():
            ae_dtype = next(ae.parameters()).dtype
            x_ae = (x * 2.0 - 1.0).to(ae_dtype)
            latents = retrieve_latents(ae.encode(x_ae), generator=self.generator)
            if hasattr(ae, "decode"):
                if "kandinsky-2" in repo_id:
                    recon = ae.decode(latents, force_not_quantize=True, return_dict=False)[0]
                else:
                    recon = ae.decode(latents, return_dict=False)[0]
            else:
                recon = ae.decoder(latents)
            recon = (recon / 2 + 0.5).clamp(0, 1).to(torch.float32)
            _, res_layers = self.lpips_model(
                x,
                recon,
                retPerLayer=True,
                normalize=True,
            )
            if self.lpips_layer >= 1:
                dist_layer = res_layers[self.lpips_layer - 1]
            elif self.lpips_layer == 0:
                dist_layer = res_layers[0]
            else:
                dist_layer = self.lpips_model(
                    x,
                    recon,
                    retPerLayer=False,
                    normalize=True,
                )
            if dist_layer.dim() > 2:
                dist_layer = dist_layer.mean(dim=(2, 3), keepdim=False)
            dist_layer = dist_layer.view(b)
            all_dist.append(dist_layer)
        if len(all_dist) > 1:
            max_dist = torch.stack(all_dist, dim=0).max(dim=0)[0]
        else:
            max_dist = all_dist[0]

        score_dist = max_dist * self.dist_sign
        score_dist = -score_dist
        signed = (score_dist - self.thr) * self.dir
        real_score = -signed * self.scale
        fake_score = signed * self.scale
        logits = torch.stack([real_score, fake_score], dim=1)
        return logits, score_dist


class FolderDataset(Dataset):
    """Load images from a directory with fixed label and optional sampling."""

    def __init__(self, root: Path, label: int, amount: int | None, data_size: int, offset: int = 0):
        self.label = label
        self.data_size = data_size
        base_ds = ImageFolder(root, amount=(amount + offset) if amount else None)
        self.items = [(Path(p), label) for p in base_ds.img_paths[offset:]]

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: Path) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            if self.data_size and self.data_size > 0:
                img = Image.new("RGB", (self.data_size, self.data_size))
            else:
                img = Image.new("RGB", (1, 1))
        if self.data_size and self.data_size > 0:
            img = img.resize((self.data_size, self.data_size), Image.BILINEAR)
        t = TF.to_tensor(img)
        return t

    def __getitem__(self, idx: int):
        p, lbl = self.items[idx]
        return self._load_image(Path(p)), lbl, str(p)


def find_best_threshold(detector: AeroBladeDetector, loader: DataLoader, device: torch.device) -> Tuple[float, str]:
    detector.eval()
    dists = []
    labels = []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            _, dist = detector(imgs)
            dists.append(dist.cpu())
            labels.append(lbls.cpu())
    dists = torch.cat(dists, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    real_vals = dists[labels == 0]
    fake_vals = dists[labels == 1]
    values = np.concatenate([real_vals, fake_vals])
    labs = np.concatenate([np.zeros_like(real_vals), np.ones_like(fake_vals)])
    uniq = np.unique(values)
    if uniq.size == 1:
        return float(uniq[0]), "ge"
    cands = []
    for i in range(len(uniq) - 1):
        cands.append((uniq[i] + uniq[i + 1]) / 2)
    cands.insert(0, uniq[0] - 1e-6)
    cands.append(uniq[-1] + 1e-6)
    best_acc = -1.0
    best_thr = cands[0]
    best_sense = "ge"
    for thr in cands:
        for sense in ("ge", "le"):
            if sense == "ge":
                preds = (values >= thr).astype(int)
            else:
                preds = (values <= thr).astype(int)
            acc = (preds == labs).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
                best_sense = sense
    print(f"    [thr] best_acc={best_acc*100:.2f}% thr={best_thr:.6f} sense={best_sense}")
    return float(best_thr), best_sense


def compute_acc(pred: torch.Tensor, label: torch.Tensor) -> Tuple[float, float, float]:
    total = label.numel()
    acc = (pred == label).sum().item() / total if total else 0.0
    real_mask = label == 0
    fake_mask = label == 1
    real_total = real_mask.sum().item()
    fake_total = fake_mask.sum().item()
    real_acc = ((pred == label) & real_mask).sum().item() / real_total if real_total else 0.0
    fake_acc = ((pred == label) & fake_mask).sum().item() / fake_total if fake_total else 0.0
    return acc, real_acc, fake_acc


def save_samples(pair_dir: Path, imgs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor, clean_pred: torch.Tensor, adv_pred: torch.Tensor, paths: List[str], save_pt: bool):
    pair_dir.mkdir(parents=True, exist_ok=True)
    for b in range(imgs.size(0)):
        label = labels[b]
        cp = clean_pred[b]
        ap = adv_pred[b]
        is_fake = bool(label.item() == 1)
        if cp != label:
            category = "clean_wrong"
        elif cp == label and ap == label:
            category = "attack_fail"
        else:
            category = "attack_success"
        cls_folder = "fake" if is_fake else "real"
        out_dir = pair_dir / category / cls_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(paths[b]).stem
        save_image(torch.clamp(adv[b], 0.0, 1.0), out_dir / f"{stem}_adv.png")
        if save_pt:
            payload = {
                "clean": torch.clamp(imgs[b].detach().cpu(), 0.0, 1.0),
                "adv": torch.clamp(adv[b].detach().cpu(), 0.0, 1.0),
                "label": int(label.item()),
                "clean_pred": int(cp.item()),
                "adv_pred": int(ap.item()),
                "path": paths[b],
            }
            torch.save(payload, out_dir / f"{stem}.pt")


def run_pair(train_gen: str, test_gen: str, cfg: argparse.Namespace, device: torch.device, detector: AeroBladeDetector) -> Dict[str, float]:
    real_val_dir = Path(cfg.data_root) / "real" / "val"
    train_val_dir = Path(cfg.data_root) / train_gen / "val"
    real_test_dir = Path(cfg.data_root) / "real" / "test"
    test_test_dir = Path(cfg.data_root) / test_gen / "test"

    if not real_val_dir.exists() or not train_val_dir.exists():
        raise FileNotFoundError(f"Missing val dirs for threshold: {real_val_dir}, {train_val_dir}")
    if not real_test_dir.exists() or not test_test_dir.exists():
        raise FileNotFoundError(f"Missing test dirs for attack: {real_test_dir}, {test_test_dir}")

    amount = cfg.amount if cfg.amount > 0 else None
    from torch.utils.data import ConcatDataset

    if cfg.thr_cache and train_gen in cfg.thr_cache:
        thr = cfg.thr_cache[train_gen]["thr"]
        sense = cfg.thr_cache[train_gen]["sense"]
        print(f"  [thr cache] use thr={thr:.6f}, sense={sense}")
        detector.set_threshold(thr, sense)
    else:
        ds_thr_real = FolderDataset(real_val_dir, label=0, amount=amount, data_size=cfg.data_size, offset=0)
        ds_thr_fake = FolderDataset(train_val_dir, label=1, amount=amount, data_size=cfg.data_size, offset=0)
        ds_thr = ConcatDataset([ds_thr_real, ds_thr_fake])
        loader_thr = DataLoader(ds_thr, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        thr, sense = find_best_threshold(detector, loader_thr, device)
        detector.set_threshold(thr, sense)

    pr = cfg.pick_real if cfg.pick_real > 0 else None
    pf = cfg.pick_fake if cfg.pick_fake > 0 else None
    ds_parts = []
    if not getattr(cfg, 'fake_only', False):
        ds_parts.append(FolderDataset(real_test_dir, label=0, amount=pr, data_size=cfg.data_size, offset=cfg.offset_real))
    if not getattr(cfg, 'real_only', False):
        ds_parts.append(FolderDataset(test_test_dir, label=1, amount=pf, data_size=cfg.data_size, offset=cfg.offset_fake))
    ds_test = ConcatDataset(ds_parts) if len(ds_parts) > 1 else ds_parts[0]
    loader_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    class AttackWrapper(nn.Module):
        def __init__(self, det: AeroBladeDetector):
            super().__init__()
            self.det = det
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits, _ = self.det(x)
            return logits
    atk_model = AttackWrapper(detector)

    attacker = AutoAttack(
        atk_model,
        norm="Linf",
        eps=cfg.eps,
        version="custom",
        attacks_to_run=["apgd-ce"],
        device=device,
    )
    if cfg.apgd_iter > 0:
        attacker.apgd.n_iter = cfg.apgd_iter
        attacker.apgd_targeted.n_iter = cfg.apgd_iter
        attacker.fab.n_iter = cfg.apgd_iter

    clean_preds = []
    adv_preds = []
    labels_all = []
    global_sample_idx = 0
    for imgs, labels, paths in loader_test:
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        detector.eval()
        with torch.no_grad():
            logits_clean, _ = detector(imgs)
            pred_clean = logits_clean.argmax(dim=1)

        batch_acc = (pred_clean == labels).float().mean().item()
        print(f"  [batch] clean_acc={batch_acc*100:.1f}% ({(pred_clean == labels).sum().item()}/{labels.size(0)})")

        save_grad_dir = str(getattr(cfg, "save_grad_probe_dir", "") or "")
        if save_grad_dir:
            Path(save_grad_dir).mkdir(parents=True, exist_ok=True)
            x_probe = imgs.detach().clone().requires_grad_(True)
            detector.zero_grad(set_to_none=True)
            logits_probe, _ = detector(x_probe)
            loss_probe = F.cross_entropy(logits_probe, labels)
            loss_probe.backward()
            grad_probe = torch.nan_to_num(x_probe.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            for b in range(imgs.size(0)):
                torch.save(
                    {
                        "grad": grad_probe[b : b + 1].detach().cpu(),
                        "input": imgs[b : b + 1].detach().cpu(),
                        "label": labels[b : b + 1].detach().cpu(),
                        "global_sample_idx": int(global_sample_idx + b),
                        "train_gen": train_gen,
                        "test_gen": test_gen,
                        "path": paths[b],
                    },
                    Path(save_grad_dir) / f"sample_{global_sample_idx + b:06d}_full_grad.pt",
                )
            detector.zero_grad(set_to_none=True)

        if getattr(cfg, "grad_probe_only", False):
            clean_preds.append(pred_clean.cpu())
            adv_preds.append(pred_clean.cpu())
            labels_all.append(labels.cpu())
            global_sample_idx += imgs.size(0)
            continue

        correct_mask = (pred_clean == labels)
        if correct_mask.sum() > 0:
            imgs_to_attack = imgs[correct_mask]
            labels_to_attack = labels[correct_mask]
            actual_aa_bs = min(cfg.aa_bs, imgs_to_attack.size(0))
            adv_imgs_subset = attacker.run_standard_evaluation(imgs_to_attack, labels_to_attack, bs=actual_aa_bs)
            adv_imgs = imgs.clone()
            adv_imgs[correct_mask] = adv_imgs_subset
        else:
            adv_imgs = imgs.clone()

        with torch.no_grad():
            logits_adv, _ = detector(adv_imgs)
            pred_adv = logits_adv.argmax(dim=1)

        clean_preds.append(pred_clean.cpu())
        adv_preds.append(pred_adv.cpu())
        labels_all.append(labels.cpu())

        if cfg.save_dir:
            pair_dir = Path(cfg.save_dir) / f"{train_gen}_on_{test_gen}"
            save_samples(pair_dir, imgs.cpu(), adv_imgs.cpu(), labels.cpu(), pred_clean.cpu(), pred_adv.cpu(), list(paths), cfg.save_pt)
        global_sample_idx += imgs.size(0)

    clean_pred = torch.cat(clean_preds, dim=0)
    adv_pred = torch.cat(adv_preds, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)

    clean_acc, clean_real_acc, clean_fake_acc = compute_acc(clean_pred, labels_cat)
    adv_acc, adv_real_acc, adv_fake_acc = compute_acc(adv_pred, labels_cat)
    return {
        "clean_acc": clean_acc,
        "clean_real_acc": clean_real_acc,
        "clean_fake_acc": clean_fake_acc,
        "adv_acc": adv_acc,
        "adv_real_acc": adv_real_acc,
        "adv_fake_acc": adv_fake_acc,
    }


def plot_heatmap(matrix: np.ndarray, gens: List[str], title: str, out_path: Path, cmap: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(gens)))
    ax.set_yticks(np.arange(len(gens)))
    ax.set_xticklabels(gens, rotation=45, ha="right")
    ax.set_yticklabels(gens)
    for i in range(len(gens)):
        for j in range(len(gens)):
            ax.text(j, i, f"{matrix[i, j]*100:.1f}", ha="center", va="center", color="black")
    ax.set_xlabel("Test generator")
    ax.set_ylabel("Train generator / classifier")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="ACC")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_thr_from_csv(thr_csv: str, gens: List[str], data_root: str, distance_metric: str) -> Dict[str, Dict[str, float]]:
    """Compute per-generator thresholds from precomputed distances CSV."""
    csv_path = Path(thr_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"thr_csv not found: {thr_csv}")
    df = pd.read_csv(csv_path)
    mask = (df["distance_metric"] == distance_metric) & (df["repo_id"] == "max")
    df = df[mask]
    def get_dist(dir_path: Path) -> np.ndarray:
        sub = df[df["dir"] == str(dir_path.resolve())]
        if sub.empty:
            raise ValueError(f"No distances for dir={dir_path} in {thr_csv}")
        return sub["distance"].to_numpy()

    real_path = Path(data_root) / "real" / "val"
    real_vals = get_dist(real_path)
    thr_map: Dict[str, Dict[str, float]] = {}
    for gen in gens:
        fake_path = Path(data_root) / gen / "val"
        fake_vals = get_dist(fake_path)
        values = np.concatenate([real_vals, fake_vals])
        labs = np.concatenate([np.zeros_like(real_vals), np.ones_like(fake_vals)])
        uniq = np.unique(values)
        cands = []
        for i in range(len(uniq) - 1):
            cands.append((uniq[i] + uniq[i + 1]) / 2)
        cands.insert(0, uniq[0] - 1e-6)
        cands.append(uniq[-1] + 1e-6)
        best_acc = -1.0
        best_thr = cands[0]
        best_sense = "ge"
        for thr in cands:
            for sense in ("ge", "le"):
                preds = (values >= thr).astype(int) if sense == "ge" else (values <= thr).astype(int)
                acc = (preds == labs).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_thr = thr
                    best_sense = sense
        thr_map[gen] = {"thr": float(best_thr), "sense": best_sense}
        print(f"[thr_csv] {gen}: acc={best_acc*100:.2f}% thr={best_thr:.6f} sense={best_sense}")
    return thr_map


def main():
    ap = argparse.ArgumentParser(description="AeroBlade cross AutoAttack (E2E recomputed reconstruction error)")
    ap.add_argument("--gens", type=str, default="adm,flux,sdv5,vqdm")
    ap.add_argument("--train_gen", type=str, default="")
    ap.add_argument("--test_gen", type=str, default="")
    ap.add_argument("--data_root", type=str, required=True, help="Dataset root (contains real/, adm/, flux/, etc.)")
    ap.add_argument("--repo_ids", type=str, nargs="+", default=[
        "CompVis/stable-diffusion-v1-1",
        "stabilityai/stable-diffusion-2-base",
        "kandinsky-community/kandinsky-2-1",
    ])
    ap.add_argument("--distance_metric", type=str, default="lpips_vgg_-1")
    ap.add_argument("--data_size", type=int, default=0, help="Input size (0 = keep original)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--aa_bs", type=int, default=1)
    ap.add_argument("--amount", type=int, default=0, help="Number of val samples for threshold calibration (0 = all)")
    ap.add_argument("--pick_real", type=int, default=0, help="Number of test real samples (0 = all)")
    ap.add_argument("--pick_fake", type=int, default=0, help="Number of test fake samples (0 = all)")
    ap.add_argument("--offset_real", type=int, default=0)
    ap.add_argument("--offset_fake", type=int, default=0)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--norm", type=str, default="Linf", choices=["Linf", "L2", "L1"])
    ap.add_argument("--apgd_iter", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--save_pt", action="store_true")
    ap.add_argument("--save_grad_probe_dir", type=str, default="")
    ap.add_argument("--grad_probe_only", action="store_true")
    ap.add_argument("--real_only", action="store_true")
    ap.add_argument("--fake_only", action="store_true")
    ap.add_argument("--out_dir", type=str, default="./aeroblade_output/cross_e2e")
    ap.add_argument("--log_json", type=str, default="cross_e2e_metrics.json")
    ap.add_argument("--thr_csv", type=str, default="", help="Path to precomputed distances CSV for threshold calibration")
    ap.add_argument("--distance_sign", type=str, choices=["pos", "neg"], default="neg",
                    help="Sign convention for distances in thr_csv (neg if stored as negative distances)")
    ap.add_argument("--debug_dist", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gens = [g.strip() for g in args.gens.split(",") if g.strip()]
    trains = [args.train_gen] if args.train_gen else gens
    tests = [args.test_gen] if args.test_gen else gens
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    detector = AeroBladeDetector(
        repo_ids=args.repo_ids,
        distance_metric=args.distance_metric,
        device=device,
        seed=args.seed,
        scale=10.0,
        dist_sign=-1.0 if args.distance_sign == "neg" else 1.0,
    ).to(device)

    thr_cache = {}
    if args.thr_csv:
        thr_cache = load_thr_from_csv(args.thr_csv, gens, args.data_root, args.distance_metric)

    clean_mat = np.zeros((len(trains), len(tests)), dtype=float)
    adv_mat = np.zeros_like(clean_mat)
    metrics: Dict[str, Dict[str, float]] = {}

    for i, train_gen in enumerate(trains):
        for j, test_gen in enumerate(tests):
            key = f"{train_gen}_on_{test_gen}"
            print(f"\n[run] {key}")
            args.thr_cache = thr_cache
            res = run_pair(train_gen, test_gen, args, device, detector)
            metrics[key] = res
            clean_mat[i, j] = res["clean_acc"]
            adv_mat[i, j] = res["adv_acc"]
            print(f"  clean acc={res['clean_acc']:.4f} (real {res['clean_real_acc']:.4f}, fake {res['clean_fake_acc']:.4f})")
            print(f"  adv   acc={res['adv_acc']:.4f} (real {res['adv_real_acc']:.4f}, fake {res['adv_fake_acc']:.4f})")

    plot_heatmap(clean_mat, trains if len(trains) > 1 else trains, "Clean ACC (AeroBlade E2E)", out_dir / "heatmap_clean_e2e.png", cmap="Reds")
    plot_heatmap(adv_mat, tests if len(tests) > 1 else tests, "Attacked ACC (AeroBlade E2E)", out_dir / "heatmap_adv_e2e.png", cmap="Blues")

    with open(out_dir / args.log_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[done] heatmaps saved to {out_dir}/, metrics saved to {out_dir / args.log_json}")


if __name__ == "__main__":
    main()
