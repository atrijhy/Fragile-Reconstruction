#!/usr/bin/env python3
# coding: utf-8
"""
purify_adversarial_samples_flux.py

FLUX.1-dev latent-space purification via flow-matching reverse Euler steps.

Approach:
  1. Encode image → latent via FLUX VAE (16ch, scaling+shift)
  2. Add flow-matching noise at sigma level:  x_t = (1-sigma)*x_0 + sigma*noise
  3. Denoise from sigma back to 0 using transformer + Euler steps
  4. Decode latent → image

Usage:
  python purify_adversarial_samples_flux.py \
    --model_path /path/to/FLUX.1-dev \
    --input_dirs "/path/to/advdir" \
    --output_root ./out \
    --t 20 \
    --batch_size 1 \
    --device cuda
"""
import argparse
import math
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="HF model folder for FLUX.1-dev")
    p.add_argument("--input_dirs", type=str, required=True)
    p.add_argument("--output_root", type=str, default="./purified_flux")
    p.add_argument("--t", type=int, default=20, help="Noise strength as integer 1..1000 (maps to sigma)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--image_size", type=int, default=512,
                   help="Input image size (must be multiple of 16 for FLUX VAE patch alignment)")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--max_files", type=int, default=-1)
    p.add_argument("--guidance_scale", type=float, default=3.5,
                   help="FLUX guidance scale (embedded as conditioning, not CFG)")
    p.add_argument("--num_inference_steps", type=int, default=28,
                   help="Total denoising steps (subset used starting from sigma_t)")
    p.add_argument("--no_pt", action="store_true", help="Do not save .pt files, only save PNG images")
    return p.parse_args()


# ---------------------------------------------------------------------------
# File collection (same as SD1.5 script)
# ---------------------------------------------------------------------------
def collect_files(input_dir: Path, recursive: bool):
    patterns = ["*_adv.png", "*_adv.pt", "*.pt"]
    for pat in patterns:
        files = sorted(input_dir.rglob(pat) if recursive else input_dir.glob(pat))
        if files:
            if pat.endswith(".png"):
                return files, "png"
            elif pat.endswith("_adv.pt"):
                return files, "pt"
            else:
                return files, "pt_single"
    files = sorted(input_dir.rglob("*.png") if recursive else input_dir.glob("*.png"))
    if files:
        return files, "png_any"
    return [], None


# ---------------------------------------------------------------------------
# FLUX Purifier
# ---------------------------------------------------------------------------
class FLUXLatentPurifier:
    def __init__(self, model_path: str, device: str = "cuda",
                 guidance_scale: float = 3.5, num_inference_steps: int = 28):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        print(f"[FLUX] Loading components from {model_path} (dtype={self.dtype}) ...")

        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers.models import FluxTransformer2DModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel
        try:
            from transformers import T5TokenizerFast
        except Exception:
            T5TokenizerFast = None

        # VAE (enable tiling to avoid OOM on large images)
        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device).eval()
        self.vae.enable_tiling()

        # Transformer
        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.dtype
        ).to(self.device).eval()

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        # Text encoders
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device).eval()

        # Try fast tokenizer first; on failure fall back to slow (use_fast=False)
        try:
            if T5TokenizerFast is None:
                raise RuntimeError("T5TokenizerFast unavailable")
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2")
        except Exception as e:
            print(f"[warn] Could not load T5TokenizerFast (falling back to slow tokenizer): {e}")
            from transformers import T5Tokenizer
            self.tokenizer_2 = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer_2", use_fast=False)
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=self.dtype
        ).to(self.device).eval()

        # Precompute empty-prompt embeddings
        self._prompt_embeds, self._pooled_prompt_embeds = self._encode_prompt("")

        # Freeze all
        for m in [self.vae, self.transformer, self.text_encoder, self.text_encoder_2]:
            for p in m.parameters():
                p.requires_grad = False

        print(f"[FLUX] Ready on {self.device}")

    @torch.no_grad()
    def _encode_prompt(self, prompt: str):
        """Encode prompt using CLIP (pooled) + T5 (sequence)."""
        # CLIP pooled
        clip_inputs = self.tokenizer(
            [prompt], padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        clip_out = self.text_encoder(clip_inputs.input_ids, output_hidden_states=False)
        pooled = clip_out.pooler_output.to(self.dtype)  # [1, 768]

        # T5 sequence
        t5_inputs = self.tokenizer_2(
            [prompt], padding="max_length", max_length=512,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        t5_out = self.text_encoder_2(t5_inputs.input_ids)
        t5_embeds = t5_out[0].to(self.dtype)  # [1, seq, 4096]

        return t5_embeds, pooled

    @torch.no_grad()
    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B,3,H,W] in [0,1] → latents."""
        x = images.to(self.device, dtype=self.dtype)
        x = x * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.sample()
        # FLUX VAE uses shift_factor and scaling_factor
        shift = self.vae.config.get("shift_factor", 0.0)
        scale = self.vae.config.get("scaling_factor", 1.0)
        latents = (latents - shift) * scale
        return latents

    @torch.no_grad()
    def _decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        shift = self.vae.config.get("shift_factor", 0.0)
        scale = self.vae.config.get("scaling_factor", 1.0)
        latents = latents / scale + shift
        images = self.vae.decode(latents.to(self.dtype)).sample
        images = (images + 1.0) / 2.0
        return images.clamp(0, 1).float()

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack latents [B, C, H, W] → [B, (H//2)*(W//2), C*4] for FLUX transformer."""
        B, C, H, W = latents.shape
        latents = latents.reshape(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)  # [B, H//2, W//2, C, 2, 2]
        latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
        return latents

    def _unpack_latents(self, packed: torch.Tensor, H: int, W: int, C: int) -> torch.Tensor:
        """Unpack [B, seq, C*4] → [B, C, H, W]."""
        B = packed.shape[0]
        packed = packed.reshape(B, H // 2, W // 2, C, 2, 2)
        packed = packed.permute(0, 3, 1, 4, 2, 5)  # [B, C, H//2, 2, W//2, 2]
        latents = packed.reshape(B, C, H, W)
        return latents

    def _prepare_image_ids(self, H: int, W: int, batch_size: int, device: torch.device):
        """Create image position IDs for rotary embeddings."""
        latent_h, latent_w = H // 2, W // 2
        ids = torch.zeros(latent_h, latent_w, 3, device=device)
        ids[..., 1] = torch.arange(latent_h, device=device)[:, None]
        ids[..., 2] = torch.arange(latent_w, device=device)[None, :]
        ids = ids.reshape(latent_h * latent_w, 3)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    @torch.no_grad()
    def purify(self, images: torch.Tensor, t: int) -> torch.Tensor:
        """
        Purify images via FLUX flow-matching.
        images: [B,3,H,W] in [0,1]
        t: noise level 1..1000
        """
        B = images.shape[0]

        # 1) Encode
        latents = self._encode_image(images)  # [B, 16, H/8, W/8]
        _, C, H, W = latents.shape

        # 2) Set up scheduler and find sigma for timestep t
        # Some FlowMatch schedulers require a `mu` argument when `use_dynamic_shifting` is True.
        # If the saved scheduler config enabled dynamic shifting but did not include `mu`,
        # call set_timesteps with a safe default (mu=0.0) and warn the user. This mirrors
        # a conservative behavior: no dynamic shifting unless explicitly specified.
        try:
            if getattr(self.scheduler.config, "use_dynamic_shifting", False):
                mu = getattr(self.scheduler.config, "mu", None)
                if mu is None:
                    # default mu once and persist it in scheduler.config to avoid noisy repeated warnings
                    mu = 0.0
                    if not getattr(self, "_warned_mu_default", False):
                        print("[warn] scheduler.use_dynamic_shifting=True but no 'mu' in config; defaulting mu=0.0")
                        self._warned_mu_default = True
                    try:
                        # persist default in scheduler config so future set_timesteps callers don't re-raise
                        setattr(self.scheduler.config, "mu", mu)
                    except Exception:
                        pass
                self.scheduler.set_timesteps(self.num_inference_steps, device=self.device, mu=mu)
            else:
                self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        except TypeError:
            # Older/newer diffusers versions may not accept mu kwarg; fall back to simple call
            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        all_timesteps = self.scheduler.timesteps  # descending
        all_sigmas = self.scheduler.sigmas

        # Map integer t (1..1000) to a fractional strength
        strength = float(t) / 1000.0
        # Find the starting index in the scheduler's timestep list
        start_idx = max(0, int(round((1.0 - strength) * len(all_timesteps))))
        if start_idx >= len(all_timesteps):
            # t is too small to add any noise — return original
            return images

        timesteps = all_timesteps[start_idx:]
        sigma_start = all_sigmas[start_idx]

        # 3) Add noise (flow matching: x_t = (1-sigma)*x_0 + sigma*noise)
        noise = torch.randn_like(latents)
        noisy = (1.0 - sigma_start) * latents + sigma_start * noise

        # 4) Prepare text conditioning
        prompt_embeds = self._prompt_embeds.expand(B, -1, -1).to(self.device, dtype=self.dtype)
        pooled_embeds = self._pooled_prompt_embeds.expand(B, -1).to(self.device, dtype=self.dtype)

        # Prepare text IDs (zeros for text sequence)
        text_ids = torch.zeros(B, prompt_embeds.shape[1], 3, device=self.device)

        # Prepare image position IDs
        image_ids = self._prepare_image_ids(H, W, B, self.device)

        # Guidance embedding
        guidance = torch.full((B,), self.guidance_scale, device=self.device, dtype=self.dtype)

        # 5) Denoise loop
        latent_model_input = noisy.to(self.dtype)

        for i, ts in enumerate(timesteps):
            # Pack latents for transformer
            packed = self._pack_latents(latent_model_input)

            # Timestep tensor — normalize to [0,1] for FLUX transformer.
            # FlowMatchEulerDiscreteScheduler.timesteps are in [0, 1000]; transformer expects [0, 1].
            t_tensor = ts.expand(B).to(self.device)
            t_max = float(all_timesteps[0])  # robust: use actual max from scheduler
            t_normalized = t_tensor / t_max if t_max > 1.0 else t_tensor

            # Transformer forward
            noise_pred = self.transformer(
                hidden_states=packed,
                timestep=t_normalized,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
                return_dict=False,
            )[0]

            # Unpack
            noise_pred = self._unpack_latents(noise_pred, H, W, C)

            # Euler step
            sigma = all_sigmas[start_idx + i]
            sigma_next = all_sigmas[start_idx + i + 1] if (start_idx + i + 1) < len(all_sigmas) else 0.0

            # Flow matching Euler: x_{t-1} = x_t + (sigma_next - sigma) * v
            # where v = noise_pred (velocity prediction)
            dt = sigma_next - sigma
            latent_model_input = latent_model_input + dt * noise_pred

        # 6) Decode
        return self._decode_latent(latent_model_input)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def pil_load_as_tensor(img: Image.Image, size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return tf(img)


def load_pt_tensor(path: Path, size: int) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        cand = data.get("adv", data.get("image", None))
        if cand is None:
            for v in data.values():
                if torch.is_tensor(v):
                    cand = v
                    break
    else:
        cand = data
    if cand is None:
        raise ValueError(f"No image tensor in {path}")
    x = cand
    if x.dim() == 4:
        x = x[0]
    if x.max() > 1.0:
        x = x.float() / 255.0
    if x.dim() == 3:
        x = F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    return x


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_directory(input_dir: str, output_root: str, purifier: FLUXLatentPurifier, args):
    input_path = Path(input_dir)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    files, ftype = collect_files(input_path, recursive=args.recursive)
    if not files:
        print(f"No files in {input_dir}")
        return 0, 0
    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"Found {len(files)} files (type: {ftype}) in {input_dir}")
    batch_size = max(1, args.batch_size)
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

    processed, failed = 0, 0
    for batch_paths in tqdm(batches, desc=f"Purifying {input_path.name}"):
        imgs, meta = [], []
        for p in batch_paths:
            try:
                if ftype in ("png", "png_any"):
                    with Image.open(p) as im:
                        img_t = pil_load_as_tensor(im.convert("RGB"), args.image_size)
                else:
                    img_t = load_pt_tensor(p, args.image_size)
                imgs.append(img_t)
                meta.append(p)
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                failed += 1

        if not imgs:
            continue

        batch_tensor = torch.stack(imgs, dim=0)
        try:
            out_imgs = purifier.purify(batch_tensor, t=args.t).cpu()
            for i, p in enumerate(meta):
                base = p.stem.replace("_adv", "")
                if args.recursive:
                    rel = p.relative_to(input_path)
                    out_dir = output_path / rel.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                else:
                    out_dir = output_path
                if not args.no_pt:
                    torch.save(
                        {"purified": out_imgs[i], "adv": batch_tensor[i]},
                        out_dir / f"{base}_purified.pt",
                    )
                save_image(out_imgs[i].unsqueeze(0), out_dir / f"{base}_purified.png")
                processed += 1
        except Exception as e:
            print(f"Failed purify batch {batch_paths[0]}: {e}")
            traceback.print_exc()
            failed += len(batch_paths)

    return processed, failed


def main():
    args = parse_args()

    # Validate image_size: must be multiple of 16 (VAE 8x downsample + patch size 2)
    if args.image_size % 16 != 0:
        old = args.image_size
        args.image_size = (args.image_size // 16) * 16
        print(f"[WARN] image_size={old} not divisible by 16, rounding to {args.image_size}")

    print("=== FLUX.1-dev Latent Purifier ===")
    print(f"Model: {args.model_path}")
    print(f"t={args.t}, batch={args.batch_size}, guidance={args.guidance_scale}")
    print(f"image_size={args.image_size}, steps={args.num_inference_steps}")
    print("==================================")

    purifier = FLUXLatentPurifier(
        model_path=args.model_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )

    total_p, total_f = 0, 0
    for d in [x.strip() for x in args.input_dirs.split(",")]:
        if not Path(d).exists():
            print(f"Warning: {d} not found, skip")
            continue
        out = Path(args.output_root) / Path(d).name
        p, f = process_directory(d, str(out), purifier, args)
        total_p += p
        total_f += f

    print(f"=== Done: {total_p} processed, {total_f} failed ===")
    print(f"Results: {args.output_root}")


if __name__ == "__main__":
    main()
