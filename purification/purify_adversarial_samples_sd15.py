#!/usr/bin/env python3
"""
Purify adversarial samples using Stable Diffusion v1.5.
Supports batch purification from multiple directories.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import time
import gc
from datetime import datetime
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str,
                   default=os.environ.get("SD15_MODEL_PATH", "/path/to/stable-diffusion-v1-5"),
                   help='Path to Stable Diffusion v1.5 model (or set SD15_MODEL_PATH env var)')
    p.add_argument('--input_dirs', type=str, required=True,
                   help='Comma-separated list of input directories containing adversarial samples')
    p.add_argument('--output_root', type=str, default='./purified_samples_sdv5',
                   help='Root directory to save purified samples')
    p.add_argument('--t', type=int, default=50,
                   help='Number of diffusion steps for purification')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--scheduler', type=str, default='ddpm',
                   choices=['ddim', 'ddpm', 'euler'],
                   help='Diffusion scheduler to use')
    p.add_argument('--recursive', action='store_true',
                   help='Recursively search for files in subdirectories')
    p.add_argument('--max_files', type=int, default=-1,
                   help='If >0, only process this many files from each directory (for debugging)')
    p.add_argument('--hang_timeout', type=int, default=600,
                   help='Per-file processing timeout in seconds; if exceeded the file is skipped')
    p.add_argument('--import_timeout', type=int, default=60,
                   help='Model import timeout in seconds')
    return p.parse_args()


def load_sd15_pipeline(model_path, scheduler_type='ddpm', device='cuda'):
    """Load Stable Diffusion v1.5 pipeline with specified scheduler."""
    print(f"Loading SD v1.5 from {model_path}...")

    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )

    # Set scheduler based on type
    if scheduler_type == 'ddim':
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_type == 'ddpm':
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_type == 'euler':
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    # Move to device and set to eval mode
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    print(f"SD v1.5 loaded with {scheduler_type.upper()} scheduler")
    return pipeline


def load_adversarial_samples(input_dir, recursive=False):
    """
    Load adversarial samples from a directory.
    Supports .pt and .png formats.
    """
    input_path = Path(input_dir)

    # Find all adversarial sample files
    glob_pattern = "**/*" if recursive else "*"

    # Prefer PNG images
    png_files = sorted(input_path.glob(f"{glob_pattern}_adv.png"))
    if png_files:
        return png_files, 'png'

    # Then try .pt files (containing clean and adv)
    pt_files = sorted(input_path.glob(f"{glob_pattern}_adv.pt"))
    if pt_files:
        return pt_files, 'pt'

    # Finally try standalone .pt files
    pt_files = sorted(input_path.glob(f"{glob_pattern}.pt"))
    if pt_files:
        return pt_files, 'pt_single'

    return [], None


def purify_image_sd15(pipeline, image_tensor, t=50, device='cuda'):
    """
    Purify a single image using SD v1.5 reverse diffusion.

    Args:
        pipeline: Loaded SD pipeline
        image_tensor: Input image tensor [C, H, W] in range [0, 1]
        t: Number of diffusion steps
        device: Device to use

    Returns:
        Purified image tensor [C, H, W] in range [0, 1]
    """
    # Convert to PIL Image (SD expects PIL images)
    # Image tensor is [C, H, W] in [0, 1], convert to [H, W, C] in [0, 255]
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)

    # Resize to SD input size (512x512)
    pil_image = pil_image.resize((512, 512), Image.LANCZOS)

    # Use empty prompt for unconditional generation
    # For purification, we want to denoise towards the data distribution
    purified_image = pipeline(
        prompt="",  # Empty prompt for unconditional
        image=pil_image,
        strength=0.8,  # How much to transform the image (0.8 = 80% denoising)
        num_inference_steps=t,
        guidance_scale=1.0,  # Unconditional guidance
        output_type="np.array"
    ).images[0]

    # Convert back to tensor [C, H, W] in [0, 1]
    purified_tensor = torch.from_numpy(purified_image).permute(2, 0, 1).float() / 255.0

    # Resize back to original size if needed (assuming 256x256 like ADM)
    if purified_tensor.shape[-1] != 256:
        purified_tensor = F.interpolate(purified_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)

    return purified_tensor


def process_directory(input_dir, output_dir, pipeline, args):
    """Process all adversarial samples in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load adversarial samples
    adv_files, file_type = load_adversarial_samples(input_dir, recursive=args.recursive)

    if not adv_files:
        print(f"Found 0 files in {input_dir}")
        return 0, 0

    if args.max_files > 0:
        adv_files = adv_files[:args.max_files]

    print(f"Found {len(adv_files)} files (type: {file_type}) in {input_dir}")

    # Transform for PNG files
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    processed = 0
    failed = 0

    for idx, adv_file in enumerate(adv_files, 1):
        start_time = time.time()

        try:
            print(f"Processing [{idx}/{len(adv_files)}]: {adv_file.name}")

            # Load adversarial sample
            if file_type == 'png':
                # PNG image file
                with Image.open(adv_file) as img:
                    img = img.convert('RGB')
                    x_adv = transform(img)
            elif file_type in ['pt', 'pt_single']:
                # PT file containing tensor data
                data = torch.load(adv_file, map_location='cpu')

                # Handle different data formats
                if isinstance(data, dict):
                    if 'adv' in data:
                        x_adv = data['adv']
                    elif 'image' in data:
                        x_adv = data['image']
                    else:
                        # Assume first tensor value
                        x_adv = list(data.values())[0]
                else:
                    x_adv = data

                # Ensure tensor is in correct format
                if x_adv.dim() == 4:  # [B, C, H, W]
                    x_adv = x_adv[0]  # Take first batch
                elif x_adv.dim() == 3:  # [C, H, W]
                    pass  # Already correct
                else:
                    raise ValueError(f"Unexpected tensor shape: {x_adv.shape}")

                # Ensure values are in [0, 1]
                if x_adv.max() > 1.0:
                    x_adv = x_adv.float() / 255.0
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Purify image
            purified_tensor = purify_image_sd15(
                pipeline, x_adv, t=args.t, device=args.device
            )

            # Save purified tensor
            rel_path = adv_file.relative_to(input_path)
            if file_type == 'png':
                # For PNG inputs, save as .pt file
                output_file = output_path / rel_path.with_suffix('.pt')
            else:
                # For PT inputs, keep same format
                output_file = output_path / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save the purified tensor
            torch.save(purified_tensor, output_file)

            processed += 1

            # Progress update
            if processed % 10 == 0:
                print(f"Processed {processed}/{len(adv_files)} files")

        except Exception as e:
            print(f"Failed to process {adv_file}: {e}")
            failed += 1

        # Check timeout
        if time.time() - start_time > args.hang_timeout:
            print(f"Timeout processing {adv_file}")
            failed += 1

    print(f"Directory {input_dir}: {processed} processed, {failed} failed")
    return processed, failed


def main():
    args = parse_args()

    print("=" * 70)
    print("Stable Diffusion v1.5 Adversarial Sample Purification")
    print("=" * 70)
    print(f"Model path:    {args.model_path}")
    print(f"Input dirs:    {args.input_dirs}")
    print(f"Output root:   {args.output_root}")
    print(f"Diffusion t:   {args.t}")
    print(f"Scheduler:     {args.scheduler}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Device:        {args.device}")
    print(f"Max files:     {args.max_files}")
    print("=" * 70)

    # Load SD v1.5 pipeline
    try:
        pipeline = load_sd15_pipeline(args.model_path, args.scheduler, args.device)
    except Exception as e:
        print(f"Failed to load SD v1.5: {e}")
        return 1

    # Process directories
    input_dirs = args.input_dirs.split(',')
    total_processed = 0
    total_failed = 0

    for input_dir in input_dirs:
        input_dir = input_dir.strip()
        if not Path(input_dir).exists():
            print(f"Warning: Input directory {input_dir} does not exist")
            continue

        # Create output directory for this input
        dir_name = Path(input_dir).name
        output_dir = Path(args.output_root) / f"t{args.t}_{args.scheduler}" / dir_name

        print(f"\nProcessing directory: {input_dir}")
        print(f"Output directory: {output_dir}")

        processed, failed = process_directory(input_dir, output_dir, pipeline, args)
        total_processed += processed
        total_failed += failed

    print("\n" + "=" * 70)
    print("Purification complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total failed:    {total_failed}")
    print(f"Results saved to: {args.output_root}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())