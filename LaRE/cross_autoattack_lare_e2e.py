#!/usr/bin/env python3
"""
Cross-evaluation for LaRE classifiers with true E2E AutoAttack.

For each train_gen -> test_gen pair:
  1. Compute clean accuracy by recomputing LaRE loss maps online.
  2. Run AutoAttack (apgd-ce), recompute loss maps on perturbed images, compute attacked accuracy.
  3. Save adversarial images to <save_dir>/<train>_on_<test>/<category>/<real|fake>/.
  4. Generate clean/attacked accuracy heatmaps.

Defaults:
  ann: <ann_root>/test_<gen>.txt
  ckpt: <ckpt_root>/ExpLaRE2_<gen>_retrain_Log_v12052017/Val_best.pth
  generators: adm, flux, sdv5, glide, vqdm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from autoattack import AutoAttack
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from transformers import CLIPTextModel, CLIPTokenizer


def _add_lare_to_path() -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent / "LaRE"]
    for c in candidates:
        if (c / "model.py").exists():
            sys.path.insert(0, str(c))
            return c
    return None


_lare_root = _add_lare_to_path()
if _lare_root is None:
    raise ImportError(
        "model.py not found. Add the LaRE directory to PYTHONPATH or pass --lare_root."
    )

import model as lare_model  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class LareE2EDataset(Dataset):
    """Load images/labels from annotation file with optional balanced sampling."""

    def __init__(
        self,
        ann_file: str,
        data_size: int,
        limit: int | None,
        clsname: str,
        pick_real: int | None = None,
        pick_fake: int | None = None,
        offset_real: int = 0,
        offset_fake: int = 0,
    ):
        self.items: List[Tuple[str, int, str]] = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label_str = line.rsplit(" ", 1)
                self.items.append((path, int(label_str), clsname))

        if pick_real or pick_fake:
            pr = pick_real or 0
            pf = pick_fake or 0
            real_cnt = fake_cnt = 0
            filtered = []
            for path, label, cn in self.items:
                if label == 0:
                    if pr is not None and pr < 0:
                        continue
                    if real_cnt < offset_real:
                        real_cnt += 1
                        continue
                    if pr == 0 or real_cnt < pr + offset_real:
                        filtered.append((path, label, cn))
                        real_cnt += 1
                else:
                    if pf is not None and pf < 0:
                        continue
                    if fake_cnt < offset_fake:
                        fake_cnt += 1
                        continue
                    if pf == 0 or fake_cnt < pf + offset_fake:
                        filtered.append((path, label, cn))
                        fake_cnt += 1
                real_done = (pr is not None and pr < 0) or (pr and pr > 0 and real_cnt >= pr + offset_real)
                fake_done = (pf is not None and pf < 0) or (pf and pf > 0 and fake_cnt >= pf + offset_fake)
                if real_done and fake_done:
                    break
            self.items = filtered
        elif limit:
            self.items = self.items[:limit]

        self.data_size = data_size

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.data_size, self.data_size))
        t = TF.to_tensor(img)
        _, h, w = t.shape
        pad_h = max(self.data_size - h, 0)
        pad_w = max(self.data_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            t = TF.pad(t, padding, fill=0)
        t = TF.center_crop(t, [self.data_size, self.data_size])
        return t

    def __getitem__(self, idx: int):
        path, label, clsname = self.items[idx]
        img = self._load_image(path)
        return img, label, clsname, path


class PromptBuilder:
    def __init__(self, prompt: str, use_template: bool):
        self.prompt = prompt
        self.use_template = use_template

    def make_prompt(self, clsname: str) -> str:
        if self.use_template:
            return self.prompt.replace("[CLS]", clsname)
        return self.prompt


class LareExtractor(nn.Module):
    """Differentiable LaRE feature extractor."""

    def __init__(
        self,
        model_path: str,
        t: int,
        ensemble_size: int,
        sd_img_size: int,
        prompt_builder: PromptBuilder,
        dtype: str,
        seed: int | None,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.t = t
        self.ensemble_size = ensemble_size
        self.sd_img_size = sd_img_size
        self.prompt_builder = prompt_builder
        self.base_seed = seed if seed is not None else 0
        torch.manual_seed(self.base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.base_seed)

        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        if dtype == "fp16":
            self.unet.half()
            self.text_encoder.half()
            self.vae.half()
        self.unet.eval()
        self.text_encoder.eval()
        self.vae.eval()
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32

    def forward(self, images: torch.Tensor, clsnames: List[str]) -> torch.Tensor:
        bsz = images.shape[0]
        imgs_sd = (images * 2.0 - 1.0).to(self.dtype)
        if self.sd_img_size > 0:
            imgs_sd = F.interpolate(imgs_sd, size=(self.sd_img_size, self.sd_img_size), mode="bilinear", align_corners=False)

        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(self.base_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.base_seed)

            latents_dist = self.vae.encode(imgs_sd).latent_dist
            eps = torch.randn_like(latents_dist.mean)
            latents = latents_dist.mean + latents_dist.std * eps
            latents = latents * self.vae.config.scaling_factor

            latents = latents.repeat_interleave(self.ensemble_size, dim=0)
            noise = torch.randn_like(latents)
            timesteps = torch.full((latents.shape[0],), self.t, device=self.device, dtype=torch.long)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            prompts = [self.prompt_builder.make_prompt(c) for c in clsnames for _ in range(self.ensemble_size)]
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
        return loss  # [B, 4, 32, 32]


class E2EClassifier(nn.Module):
    """Wrapper that computes loss maps online before classification."""

    def __init__(self, base_model: nn.Module, extractor: LareExtractor):
        super().__init__()
        self.base_model = base_model
        self.extractor = extractor
        self.classnames: List[str] | None = None
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)

    def set_classnames(self, clsnames: List[str]) -> None:
        self.classnames = clsnames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classnames is None:
            raise RuntimeError("set_classnames() must be called before forward")
        loss_maps = self.extractor(x, self.classnames)
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm, loss_maps)


def load_classifier(model_name: str, clip_type: str, ckpt_path: str, device: torch.device) -> nn.Module:
    ctor = getattr(lare_model, model_name)
    model = ctor(num_class=2, clip_type=clip_type)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


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


def run_attack_with_fallback(
    attacker: AutoAttack,
    wrapper: E2EClassifier,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    clsnames: List[str],
    aa_bs: int,
) -> torch.Tensor:
    """Run attack on batch; fall back to per-sample if dimension mismatch occurs."""
    try:
        wrapper.set_classnames(list(clsnames))
        return attacker.run_standard_evaluation(imgs, labels, bs=aa_bs)
    except RuntimeError as e:
        if "size of tensor a" in str(e) or "size mismatch" in str(e) or "invalid for input of size" in str(e):
            print(f"[warn] batch attack failed ({e}); fallback to per-sample")
            adv_list = []
            for b in range(imgs.size(0)):
                img_b = imgs[b : b + 1]
                label_b = labels[b : b + 1]
                cls_b = [clsnames[b]]
                wrapper.set_classnames(cls_b)
                adv_b = attacker.run_standard_evaluation(img_b, label_b, bs=aa_bs)
                adv_list.append(adv_b)
            return torch.cat(adv_list, dim=0)
        else:
            raise


def run_pair(train_gen: str, test_gen: str, cfg: argparse.Namespace, device: torch.device, extractor: LareExtractor) -> Dict[str, float]:
    ann_file = str(Path(cfg.ann_root) / f"test_{test_gen}.txt")
    ckpt_path = str(Path(cfg.ckpt_root) / cfg.ckpt_pattern.format(gen=train_gen))

    pr = cfg.pick_real if cfg.pick_real > 0 else None
    pf = cfg.pick_fake if cfg.pick_fake > 0 else None
    if getattr(cfg, 'fake_only', False):
        pr = -1
    if getattr(cfg, 'real_only', False):
        pf = -1

    dataset = LareE2EDataset(
        ann_file=ann_file,
        data_size=cfg.data_size,
        limit=(cfg.limit if cfg.limit > 0 else None),
        clsname=test_gen,
        pick_real=pr,
        pick_fake=pf,
        offset_real=cfg.offset_real,
        offset_fake=cfg.offset_fake,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    base_classifier = load_classifier(cfg.model, cfg.clip_type, ckpt_path, device)
    wrapper = E2EClassifier(base_classifier, extractor).to(device)

    attacker = AutoAttack(
        wrapper,
        norm=cfg.norm,
        eps=cfg.eps,
        version="custom",
        attacks_to_run=["apgd-ce"],
        device=device,
        seed=cfg.seed,
    )
    if cfg.apgd_iter > 0:
        attacker.apgd.n_iter = cfg.apgd_iter
        attacker.apgd_targeted.n_iter = cfg.apgd_iter
        attacker.fab.n_iter = cfg.apgd_iter

    clean_preds = []
    clean_labels = []
    adv_preds = []
    adv_labels = []
    global_sample_idx = 0

    pair_save_dir = None
    if cfg.save_dir:
        pair_save_dir = Path(cfg.save_dir) / f"{train_gen}_on_{test_gen}"
        pair_save_dir.mkdir(parents=True, exist_ok=True)

    for imgs, labels, clsnames, paths in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        wrapper.set_classnames(list(clsnames))
        with torch.no_grad():
            clean_logits = wrapper(imgs)
            clean_pred = clean_logits.argmax(dim=1)

        batch_acc = (clean_pred == labels).float().mean().item()
        print(f"  [batch] clean_acc={batch_acc*100:.1f}% ({(clean_pred == labels).sum().item()}/{labels.size(0)})")

        save_grad_dir = str(getattr(cfg, "save_grad_probe_dir", "") or "")
        if save_grad_dir:
            Path(save_grad_dir).mkdir(parents=True, exist_ok=True)
            wrapper.set_classnames(list(clsnames))
            x_probe = imgs.detach().clone().requires_grad_(True)
            wrapper.zero_grad(set_to_none=True)
            logits_probe = wrapper(x_probe)
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
            wrapper.zero_grad(set_to_none=True)

        if getattr(cfg, "grad_probe_only", False):
            clean_preds.append(clean_pred.cpu())
            clean_labels.append(labels.cpu())
            adv_preds.append(clean_pred.cpu())
            adv_labels.append(labels.cpu())
            global_sample_idx += imgs.size(0)
            continue

        correct_mask = (clean_pred == labels)
        if correct_mask.sum() > 0:
            imgs_to_attack = imgs[correct_mask]
            labels_to_attack = labels[correct_mask]
            clsnames_to_attack = [cls for idx, cls in enumerate(clsnames) if correct_mask[idx]]
            actual_aa_bs = min(cfg.aa_bs, imgs_to_attack.size(0))
            adv_imgs_subset = run_attack_with_fallback(attacker, wrapper, imgs_to_attack, labels_to_attack, clsnames_to_attack, actual_aa_bs)
            adv_imgs = imgs.clone()
            adv_imgs[correct_mask] = adv_imgs_subset
        else:
            adv_imgs = imgs.clone()

        wrapper.set_classnames(list(clsnames))
        with torch.no_grad():
            adv_logits = wrapper(adv_imgs)
            adv_pred = adv_logits.argmax(dim=1)

        clean_preds.append(clean_pred.cpu())
        clean_labels.append(labels.cpu())
        adv_preds.append(adv_pred.cpu())
        adv_labels.append(labels.cpu())

        if pair_save_dir:
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
                out_dir = pair_save_dir / category / cls_folder
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(paths[b]).stem
                save_image(torch.clamp(adv_imgs[b], 0.0, 1.0), out_dir / f"{stem}_adv.png")
                if cfg.save_pt:
                    payload = {
                        "clean": torch.clamp(imgs[b].detach().cpu(), 0.0, 1.0),
                        "adv": torch.clamp(adv_imgs[b].detach().cpu(), 0.0, 1.0),
                        "label": int(label.item()),
                        "clean_pred": int(cp.item()),
                        "adv_pred": int(ap.item()),
                    }
                    torch.save(payload, out_dir / f"{stem}.pt")
        global_sample_idx += imgs.size(0)

    clean_pred = torch.cat(clean_preds, dim=0)
    clean_label = torch.cat(clean_labels, dim=0)
    adv_pred = torch.cat(adv_preds, dim=0)
    adv_label = torch.cat(adv_labels, dim=0)

    clean_acc, clean_real_acc, clean_fake_acc = compute_acc(clean_pred, clean_label)
    adv_acc, adv_real_acc, adv_fake_acc = compute_acc(adv_pred, adv_label)
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


def main():
    ap = argparse.ArgumentParser(description="LaRE cross AutoAttack E2E (recompute loss map)")
    ap.add_argument("--gens", type=str, default="adm,flux,sdv5,glide,vqdm")
    ap.add_argument("--train_gen", type=str, default="")
    ap.add_argument("--test_gen", type=str, default="")
    ap.add_argument("--ann_root", type=str, required=True, help="Directory containing test_<gen>.txt annotation files")
    ap.add_argument("--ckpt_root", type=str, required=True, help="Root directory for LaRE checkpoints")
    ap.add_argument("--ckpt_pattern", type=str, default="ExpLaRE2_{gen}_retrain_Log_v12052017/Val_best.pth")
    ap.add_argument("--model", type=str, default="CLipClassifierWMapV6")
    ap.add_argument("--clip_type", type=str, default="RN50")
    ap.add_argument("--data_size", type=int, default=224)
    ap.add_argument("--sd_img_size", type=int, default=256)
    ap.add_argument("--t", type=int, default=200)
    ap.add_argument("--ensemble_size", type=int, default=4)
    ap.add_argument("--prompt", type=str, default="a photo")
    ap.add_argument("--use_prompt_template", action="store_true")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--aa_bs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="Total sample cap (0 = unlimited)")
    ap.add_argument("--pick_real", type=int, default=0, help="Number of real samples to use (0 = unlimited)")
    ap.add_argument("--pick_fake", type=int, default=0, help="Number of fake samples to use (0 = unlimited)")
    ap.add_argument("--offset_real", type=int, default=0)
    ap.add_argument("--offset_fake", type=int, default=0)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--norm", type=str, default="Linf", choices=["Linf", "L2", "L1"])
    ap.add_argument("--apgd_iter", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--save_pt", action="store_true")
    ap.add_argument("--save_grad_probe_dir", type=str, default="")
    ap.add_argument("--grad_probe_only", action="store_true")
    ap.add_argument("--real_only", action="store_true")
    ap.add_argument("--fake_only", action="store_true")
    ap.add_argument("--out_dir", type=str, default="./lare_adv_outputs/cross_e2e")
    ap.add_argument("--log_json", type=str, default="cross_e2e_metrics.json")
    ap.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                    help="Path to SD 1.5 model folder (HuggingFace format)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gens = [g.strip() for g in args.gens.split(",") if g.strip()]
    trains = [args.train_gen] if args.train_gen else gens
    tests = [args.test_gen] if args.test_gen else gens

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    prompt_builder = PromptBuilder(args.prompt, args.use_prompt_template)
    extractor = LareExtractor(
        model_path=args.pretrained_model_name_or_path,
        t=args.t,
        ensemble_size=args.ensemble_size,
        sd_img_size=args.sd_img_size,
        prompt_builder=prompt_builder,
        dtype=args.dtype,
        seed=args.seed,
        device=device,
    ).to(device)

    clean_matrix = np.zeros((len(trains), len(tests)), dtype=float)
    adv_matrix = np.zeros_like(clean_matrix)
    metrics: Dict[str, Dict[str, float]] = {}
    for i, train_gen in enumerate(trains):
        for j, test_gen in enumerate(tests):
            key = f"{train_gen}_on_{test_gen}"
            print(f"\n[run E2E] {key}")
            res = run_pair(train_gen, test_gen, args, device, extractor)
            metrics[key] = res
            clean_matrix[i, j] = res["clean_acc"]
            adv_matrix[i, j] = res["adv_acc"]
            print(f"  clean acc={res['clean_acc']:.4f} (real {res['clean_real_acc']:.4f}, fake {res['clean_fake_acc']:.4f})")
            print(f"  adv   acc={res['adv_acc']:.4f} (real {res['adv_real_acc']:.4f}, fake {res['adv_fake_acc']:.4f})")

    plot_heatmap(clean_matrix, trains if len(trains) > 1 else trains, "Clean ACC (E2E)", out_dir / "heatmap_clean_e2e.png", cmap="Reds")
    plot_heatmap(adv_matrix, tests if len(tests) > 1 else tests, "Attacked ACC (E2E)", out_dir / "heatmap_adv_e2e.png", cmap="Blues")

    with open(out_dir / args.log_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[done] heatmaps saved to {out_dir}/, metrics saved to {out_dir / args.log_json}")


if __name__ == "__main__":
    main()
