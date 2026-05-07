#!/usr/bin/env python3
# coding: utf-8
"""
purify_adversarial_samples_vqdm.py

VQ-Diffusion (ITHQ) purification via discrete token masking and denoising.

Uses the official VQDiffusionPipeline from diffusers for compatibility.

Approach:
  1. Encode image → VQ tokens (32x32 grid, codebook size 4096)
  2. At timestep t, corrupt tokens by masking a fraction of them
     (replace with MASK token = num_embeddings = 4096)
  3. Denoise: use pipeline's transformer to predict original tokens iteratively
  4. Decode VQ tokens → image

Usage:
  python purify_adversarial_samples_vqdm.py \
    --model_path /path/to/vq-diffusion-ithq \
    --input_dirs "/path/to/advdir" \
    --output_root ./out \
    --t 20 \
    --batch_size 1 \
    --device cuda
"""
import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="HF model folder for VQ-Diffusion")
    p.add_argument("--input_dirs", type=str, required=True)
    p.add_argument("--output_root", type=str, default="./purified_vqdm")
    p.add_argument("--t", type=int, default=20, help="Noise timestep 1..100 (VQ-Diffusion has 100 steps)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--image_size", type=int, default=256,
                   help="Input image size (VQ-Diffusion ITHQ uses 256x256)")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--max_files", type=int, default=-1)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=100,
                   help="Number of denoising steps (default: 100)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# File collection
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
# VQ-Diffusion Purifier using Official Pipeline
# ---------------------------------------------------------------------------
class VQDMPurifier:
    def __init__(self, model_path: str, device: str = "cuda",
                 guidance_scale: float = 5.0, num_inference_steps: int = 100):
        self.device = torch.device(device)
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        print(f"[VQDM] Loading VQ-Diffusion components from {model_path} ...")

        from diffusers import VQModel, VQDiffusionScheduler
        from diffusers.models import Transformer2DModel
        from transformers import CLIPTextModel, CLIPTokenizer

        # Load components manually to avoid deprecated pipeline issues
        self.vqvae = VQModel.from_pretrained(
            model_path, subfolder="vqvae"
        ).to(self.device).eval()

        self.transformer = Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer"
        ).to(self.device).eval()

        # CRITICAL FIX: VQ-Diffusion models trained with old diffusers need chunk_dim=1
        # not chunk_dim=0. This is a compatibility fix for diffusers 0.35+ with old models.
        for block in self.transformer.transformer_blocks:
            if hasattr(block, 'norm1') and hasattr(block.norm1, 'chunk_dim'):
                block.norm1.chunk_dim = 1
            if hasattr(block, 'norm2') and hasattr(block.norm2, 'chunk_dim'):
                block.norm2.chunk_dim = 1
            if hasattr(block, 'norm3') and hasattr(block.norm3, 'chunk_dim'):
                block.norm3.chunk_dim = 1

        self.scheduler = VQDiffusionScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder"
        ).to(self.device).eval()

        # Config
        self.num_embeddings = self.vqvae.config.num_vq_embeddings  # 4096
        self.mask_token = self.num_embeddings  # 4096 = MASK
        self.num_timesteps = self.scheduler.config.num_train_timesteps  # 100
        self.latent_h = self.latent_w = 32  # VQ-Diffusion ITHQ: 256/8 = 32

        # Precompute empty-prompt embedding
        self._uncond_embeds = self._encode_text("")

        # Freeze all
        for p in self.vqvae.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        print(f"[VQDM] Ready on {self.device}, codebook={self.num_embeddings}, "
              f"mask_token={self.mask_token}, timesteps={self.num_timesteps}")

    @torch.no_grad()
    def _encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        inputs = self.tokenizer(
            [prompt], padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.text_encoder(inputs.input_ids)
        return out.last_hidden_state  # [1, seq, dim]

    @torch.no_grad()
    def _encode_to_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to VQ tokens.
        images: [B,3,H,W] in [0,1] → token indices [B, h*w].
        """
        x = images.to(self.device)
        x = x * 2.0 - 1.0  # [0,1] -> [-1, 1]

        # VQ-VAE encode
        h = self.vqvae.encoder(x)
        h = self.vqvae.quant_conv(h)

        # Vector quantize
        _, _, (_, _, min_encoding_indices) = self.vqvae.quantize(h)

        # Reshape to [B, h*w]
        B = images.shape[0]
        tokens = min_encoding_indices.reshape(B, -1)  # [B, 1024]
        return tokens

    @torch.no_grad()
    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode VQ tokens to images.
        tokens: [B, h*w] → images [B,3,H,W] in [0,1].
        """
        B = tokens.shape[0]

        # Look up embeddings from codebook
        tokens_flat = tokens.reshape(-1).long()
        embeddings = self.vqvae.quantize.embedding(tokens_flat)  # [B*h*w, embed_dim]
        embed_dim = embeddings.shape[-1]

        # Reshape to spatial dimensions
        embeddings = embeddings.reshape(B, self.latent_h, self.latent_w, embed_dim)
        embeddings = embeddings.permute(0, 3, 1, 2)  # [B, embed_dim, h, w]

        # Post-quant conv and decode
        embeddings = self.vqvae.post_quant_conv(embeddings)
        images = self.vqvae.decoder(embeddings)

        # Convert from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        return images.clamp(0, 1)

    @torch.no_grad()
    def _mask_tokens(self, tokens: torch.Tensor, t: int) -> torch.Tensor:
        """
        Mask a fraction of tokens at timestep t.
        Higher t → more masking (more noise).
        Uses the scheduler's cumulative gamma schedule.
        """
        B, N = tokens.shape
        t_idx = min(max(int(t) - 1, 0), self.num_timesteps - 1)

        # Use scheduler's gamma schedule for mask probability
        # ctt = cumulative gamma = probability of being in MASK state at timestep t
        # Higher t → higher ctt → more masking
        if hasattr(self.scheduler, 'log_cumprod_ct'):
            ctt = self.scheduler.log_cumprod_ct[t_idx].exp().item()
            mask_prob = ctt  # ctt IS the mask probability (not 1-ctt!)
        else:
            # Fallback: linear interpolation from config
            gamma_start = self.scheduler.config.gamma_cum_start
            gamma_end = self.scheduler.config.gamma_cum_end
            frac = float(t_idx) / max(float(self.num_timesteps - 1), 1.0)
            mask_prob = gamma_start + (gamma_end - gamma_start) * frac

        # Generate random mask
        mask = torch.rand(B, N, device=tokens.device) < mask_prob
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token
        num_masked = mask.sum().item()
        print(f"[VQDM] t={t}, t_idx={t_idx}, mask_prob={mask_prob:.4f}, "
              f"masked={num_masked}/{B*N} ({100*num_masked/(B*N):.1f}%)")
        return masked_tokens

    @torch.no_grad()
    def purify(self, images: torch.Tensor, t: int) -> torch.Tensor:
        """
        Purify images via VQ-Diffusion discrete denoising.

        Args:
            images: [B,3,H,W] in [0,1]
            t: noise level 1..100 (higher = more noise)

        Returns:
            Purified images [B,3,H,W] in [0,1]
        """
        B = images.shape[0]

        # 1) Encode to VQ tokens
        tokens = self._encode_to_tokens(images)  # [B, 1024]

        # 2) Add noise: mask tokens at timestep t
        noisy_tokens = self._mask_tokens(tokens, t)

        # 3) Prepare text conditioning (use empty prompt for purification)
        prompt_embeds = self._uncond_embeds.expand(B, -1, -1)

        # 4) Set up scheduler for denoising
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Find starting timestep index
        # We only denoise from t downward to 0
        t_idx_target = min(max(int(t) - 1, 0), self.num_timesteps - 1)
        start_idx = 0
        for i, ts in enumerate(timesteps):
            if ts.item() <= t_idx_target:
                start_idx = i
                break

        # 5) Iterative denoising from t down to 1
        # NOTE: Skip t=0 because the scheduler at t=0 bypasses the posterior
        # and directly uses model_output. The Transformer2DModel in newer diffusers
        # has compatibility issues that produce incorrect logits, so the posterior
        # (which blends model prediction with current state) works but direct
        # prediction does not. At t=1, all MASK tokens are already resolved.
        sample = noisy_tokens
        for i in range(start_idx, len(timesteps)):
            timestep = timesteps[i]

            # Skip t=0: scheduler uses raw model output (no posterior),
            # which corrupts all tokens due to transformer compat issues
            if timestep.item() == 0:
                break

            # Expand timestep for batch
            timestep_tensor = timestep.expand(B).to(self.device)

            # Predict the denoised tokens (logits over codebook)
            model_output = self.transformer(
                sample,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep_tensor,
            ).sample

            # Normalize logits to valid log probabilities
            # (matches official pipeline's logsumexp normalization)
            model_output -= torch.logsumexp(model_output, dim=1, keepdim=True)

            # Truncate and clamp logits (following official pipeline)
            model_output = self._truncate(model_output, truncation_rate=1.0)
            model_output = model_output.clamp(-70)

            # Scheduler step: predict previous sample (less noisy tokens)
            sample = self.scheduler.step(
                model_output,
                timestep=timestep,
                sample=sample,
                generator=None,
            ).prev_sample

        # 6) Decode final tokens to image
        # Clamp to valid codebook range (remove any remaining MASK tokens)
        sample = sample.clamp(0, self.num_embeddings - 1)
        purified_images = self._decode_tokens(sample)
        return purified_images

    def _truncate(self, model_output: torch.Tensor, truncation_rate: float) -> torch.Tensor:
        """
        Truncate predicted logits to top-k with cumulative probability <= truncation_rate.
        Following official VQDiffusionPipeline implementation.
        """
        if truncation_rate == 1.0:
            return model_output

        # model_output: [B, num_classes, H*W]
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(model_output, dim=1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)

        # Compute cumulative probabilities
        cum_probs = torch.cumsum(sorted_probs, dim=1)

        # Find cutoff: where cumulative probability exceeds truncation_rate
        sorted_probs[cum_probs > truncation_rate] = 0.0

        # Scatter back to original indices
        probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)

        # Convert back to logits
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        truncated_output = torch.log(probs.clamp(min=1e-10))

        return truncated_output


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
def process_directory(input_dir: str, output_root: str, purifier: VQDMPurifier, args):
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
                save_image(out_imgs[i].unsqueeze(0), out_dir / f"{base}_purified.png")
                processed += 1
        except Exception as e:
            print(f"Failed purify batch {batch_paths[0]}: {e}")
            traceback.print_exc()
            failed += len(batch_paths)

    return processed, failed


def main():
    args = parse_args()
    print("=== VQ-Diffusion (ITHQ) Purifier ===")
    print(f"Model: {args.model_path}")
    print(f"t={args.t}, batch={args.batch_size}, guidance={args.guidance_scale}")
    print(f"Image size: {args.image_size} (VQ tokens: {args.image_size // 8}x{args.image_size // 8})")
    print(f"Inference steps: {args.num_inference_steps}")
    print("=====================================")

    purifier = VQDMPurifier(
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
