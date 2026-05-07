#!/usr/bin/env python3
"""
Purify adversarial samples using DiffPure.
Supports batch purification from multiple input directories.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import torchvision.transforms as transforms
import time
import gc
from datetime import datetime
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent / 'DiffPure'))

# NOTE: heavy DiffPure imports moved into build_purifier() to allow
# early logging and to avoid blocking at module import time.



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='imagenet.yml',
                   help='DiffPure config file (in DiffPure/configs/)')
    p.add_argument('--diffusion_model_path', type=str, required=True,
                   help='Path to diffusion model checkpoint')
    p.add_argument('--input_dirs', type=str, required=True,
                   help='Comma-separated list of input directories containing adversarial samples')
    p.add_argument('--output_root', type=str, default='./purified_samples',
                   help='Root directory to save purified samples')
    p.add_argument('--t', type=int, default=100,
                   help='Number of diffusion steps for purification')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--diffusion_type', type=str, default='ode',
                   choices=['ode', 'ddpm', 'ddim', 'sde'])
    p.add_argument('--recursive', action='store_true',
                   help='Recursively search for .pt files in subdirectories')
    p.add_argument('--max_files', type=int, default=-1,
                   help='If >0, only process this many files from each directory (for debugging)')
    p.add_argument('--hang_timeout', type=int, default=600,
                   help='Per-file processing timeout in seconds; if exceeded the file is skipped')
    p.add_argument('--import_timeout', type=int, default=60,
                   help='Timeout in seconds for testing DiffPure module imports (diagnostic)')
    p.add_argument('--no_pt', action='store_true',
                   help='Do not save .pt files, only save PNG images')
    return p.parse_args()


def build_purifier(args):
    """Build DiffPure purifier."""
    print("[1/5] Loading config...")
    cfg_path = Path(__file__).parent / 'DiffPure' / 'configs' / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    # dict2namespace is in DiffPure/utils.py - import it here (lazy) so
    # we can detect import errors separately from heavy model imports.
    try:
        from utils import dict2namespace
    except Exception as e:
        raise RuntimeError(f"Failed to import dict2namespace: {e}")

    config = dict2namespace(cfg_yaml)

    device = torch.device(args.device)
    config.device = device
    print(f"[2/5] Config loaded, device: {device}")

    diffusion_args = argparse.Namespace(
        diffusion_model_path=args.diffusion_model_path,
        score_type='guided_diffusion',
        t=args.t,
        sample_step=1,
        step_size=2e-2,
        seed=42,
        fix_rand=True,
        log_dir='/tmp/purify_log'
    )

    # Some runners expect additional arguments (rand_t, t_delta, use_bm etc.).
    # Provide sensible defaults here so the runner can read them from the namespace.
    if not hasattr(diffusion_args, 'rand_t'):
        setattr(diffusion_args, 'rand_t', False)
    if not hasattr(diffusion_args, 't_delta'):
        setattr(diffusion_args, 't_delta', 0)
    if not hasattr(diffusion_args, 'use_bm'):
        setattr(diffusion_args, 'use_bm', False)

    print(f"[3/5] Diffusion method: {args.diffusion_type.upper()}")
    print("[4/5] Loading diffusion model (may take 1-2 minutes)...")
    print(f"      model path: {args.diffusion_model_path}")

    start_time = time.time()
    # Import heavy modules lazily so we can trace where a hang occurs
    print(f"[4.1/5] Testing DiffPure runner imports... {datetime.now().isoformat()}", flush=True)

    # Diagnostic: try importing each module in a short-lived subprocess to
    # determine which import (if any) blocks or crashes. This avoids hanging
    # the main process and gives a clearer error message.
    diffpure_path = str(Path(__file__).parent / 'DiffPure')
    # Only test the modules actually needed for the selected diffusion type.
    # Importing all runners unconditionally can trigger heavy torch/hipify/cpp_extension
    # imports (slow or blocking). Limit tests to the required runner + utils.
    modules_to_test = ['utils']
    if args.diffusion_type == 'ode':
        modules_to_test.insert(0, 'runners.diffpure_ode')
    elif args.diffusion_type == 'ddpm':
        modules_to_test.insert(0, 'runners.diffpure_guided')
    elif args.diffusion_type == 'ddim':
        modules_to_test.insert(0, 'runners.diffpure_guided_ddim')
    elif args.diffusion_type == 'sde':
        modules_to_test.insert(0, 'runners.diffpure_sde')

    def test_import_verbose(module_name, timeout):
        """Test module import in subprocess. Returns (returncode, output_text) or (None, msg) on timeout."""
        logs_dir = Path(__file__).parent / 'DiffPure' / 'import_logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = module_name.replace('.', '_')
        log_file = logs_dir / f'{safe_name}.log'

        code = (
            f"import sys, traceback, os\n"
            f"sys.path.insert(0, '{diffpure_path}')\n"
            f"print('TEST_IMPORT_START:{module_name}', flush=True)\n"
            f"try:\n"
            f"    import {module_name}\n"
            f"    print('TEST_IMPORT_OK:{module_name}', flush=True)\n"
            f"except Exception as e:\n"
            f"    print('TEST_IMPORT_EXC:{module_name}', e, flush=True)\n"
            f"    traceback.print_exc()\n"
        )

        try:
            proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=timeout)
            out_text = proc.stdout + proc.stderr
            # write full output to log for later inspection
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(out_text)
            except Exception:
                pass
            return proc.returncode, out_text
        except subprocess.TimeoutExpired:
            msg = f"TIMEOUT after {timeout}s when importing {module_name}. Log: {log_file}"
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(msg + '\n')
            except Exception:
                pass
            return None, msg

    for m in modules_to_test:
        # give SDE more time because its imports may trigger CPP/CUDA checks
        to = args.import_timeout * 5 if 'sde' in m else args.import_timeout
        print(f"[4.1.x] Testing import: {m} (timeout={to}s)")
        rc, out = test_import_verbose(m, to)
        if rc is None:
            raise RuntimeError(out)
        if rc != 0:
            # If import failed, point to the per-module log file for debugging
            safe_name = m.replace('.', '_')
            log_file = Path(__file__).parent / 'DiffPure' / 'import_logs' / f"{safe_name}.log"
            raise RuntimeError(f"Module {m} import failed (rc={rc}). Log: {log_file}\n\n{out}")
        print(f"  -> {m} import OK (full log: DiffPure/import_logs/{m.replace('.', '_')}.log)")

    print(f"[4.2/5] DiffPure imports OK - {datetime.now().isoformat()}", flush=True)

    # Now import in-process (should be fast since tests passed)
    try:
        # Import only the runner required for the chosen diffusion_type to avoid
        # pulling in heavy modules for unused runners (which can trigger
        # torch cpp_extension/hipify initialization and be very slow).
        from utils import dict2namespace
        if args.diffusion_type == 'ode':
            from runners.diffpure_ode import OdeGuidedDiffusion
            PurRunner = OdeGuidedDiffusion
        elif args.diffusion_type == 'ddpm':
            from runners.diffpure_guided import GuidedDiffusion
            PurRunner = GuidedDiffusion
        elif args.diffusion_type == 'ddim':
            from runners.diffpure_guided_ddim import GuidedDiffusionDDIM
            PurRunner = GuidedDiffusionDDIM
        elif args.diffusion_type == 'sde':
            from runners.diffpure_sde import RevGuidedDiffusion
            PurRunner = RevGuidedDiffusion
        else:
            raise NotImplementedError(f"Diffusion type {args.diffusion_type} not implemented")
    except Exception as e:
        raise RuntimeError(f"DiffPure runner import failed: {e}")

    print(f"[4.3/5] DiffPure in-process imports done - {datetime.now().isoformat()}", flush=True)

    print(f"[4.4/5] Building purifier (loading weights) - {datetime.now().isoformat()}", flush=True)
    # Construct the purifier using the runner class we imported above.
    purifier = PurRunner(diffusion_args, config, device=device)

    elapsed = time.time() - start_time
    print(f"[5/5] Model loaded in {elapsed:.1f}s")
    return purifier, config


def load_adversarial_samples(input_dir, recursive=False):
    """Load adversarial samples from directory. Supports .pt and .png formats."""
    input_path = Path(input_dir)

    glob_pattern = "**/*" if recursive else "*"

    png_files = sorted(input_path.glob(f"{glob_pattern}_adv.png"))
    if png_files:
        return png_files, 'png'

    pt_files = sorted(input_path.glob(f"{glob_pattern}_adv.pt"))
    if pt_files:
        return pt_files, 'pt'

    pt_files = sorted(input_path.glob(f"{glob_pattern}.pt"))
    if pt_files:
        return pt_files, 'pt_single'

    return [], None


def purify_batch(purifier, x_adv, args, config, batch_idx=0):
    """
    Purify a batch of adversarial samples using DiffPure.

    Args:
        purifier: DiffPure purifier instance
        x_adv: adversarial tensor [B, 3, H, W], values in [0, 1]
        args: parsed arguments
        config: DiffPure config
        batch_idx: batch index (avoids file conflicts)

    Returns:
        x_purified: purified tensor [B, 3, H, W], values in [0, 1]
    """
    if 'imagenet' in args.config.lower():
        x_adv = F.interpolate(x_adv, size=(256, 256), mode='bilinear', align_corners=False)

    x_adv_scaled = (x_adv - 0.5) * 2.0

    with torch.no_grad():
        x_purified = purifier.image_editing_sample(
            x_adv_scaled,
            bs_id=batch_idx,
            tag=f'purify_{batch_idx}'
        )

    x_purified = (x_purified + 1.0) * 0.5

    if 'imagenet' in args.config.lower() and x_purified.shape[-1] != x_adv.shape[-1]:
        x_purified = F.interpolate(x_purified, size=(256, 256), mode='bilinear', align_corners=False)

    return x_purified


def process_directory(input_dir, output_dir, purifier, args, config):
    """Process all adversarial samples in a directory."""
    print(f"\nProcessing directory: {input_dir}")

    adv_files, file_type = load_adversarial_samples(input_dir, recursive=args.recursive)

    if not adv_files:
        print("  No adversarial samples found, skipping")
        return

    total_files = len(adv_files)
    print(f"  Found {total_files} files (type: {file_type})")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    input_path = Path(input_dir)

    print(f"\nPurifying {total_files} samples...")

    # Transform (for PNG files)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    start_time = time.time()
    success_count = 0
    error_count = 0

    for idx, adv_file in enumerate(adv_files, 1):
        # Respect --max_files for debugging
        if args.max_files > 0 and idx > args.max_files:
            print(f"  -- Reached max_files={args.max_files}, stopping")
            break
        try:
            iter_start = time.time()

            print(f"\n  [{idx}/{total_files}] Processing: {adv_file.name} - {datetime.now().isoformat()}", flush=True)

            if file_type == 'png':
                try:
                    with Image.open(adv_file) as img:
                        img = img.convert('RGB')
                        x_adv = transform(img)
                except Exception as e:
                    raise RuntimeError(f"Cannot open PNG {adv_file}: {e}")
                x_clean = None

            elif file_type == 'pt':
                data = torch.load(adv_file, map_location='cpu')
                if isinstance(data, dict):
                    x_adv = data.get('adv', data.get('adversarial'))
                    x_clean = data.get('clean', None)
                else:
                    x_adv = data
                    x_clean = None
            elif file_type == 'pt_single':
                x_adv = torch.load(adv_file, map_location='cpu')
                x_clean = None
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            if x_adv.dim() == 3:
                x_adv = x_adv.unsqueeze(0)

            x_adv = x_adv.to(device)

            print(f"    -> Starting purification - {datetime.now().isoformat()}", flush=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            purify_start = time.time()
            x_purified = purify_batch(purifier, x_adv, args, config, batch_idx=idx+100)
            purify_elapsed = time.time() - purify_start
            print(f"    <- Purification done in {purify_elapsed:.1f}s - {datetime.now().isoformat()}", flush=True)

            if purify_elapsed > args.hang_timeout:
                print(f"    WARNING: {adv_file.name} purification took {purify_elapsed:.1f}s > hang_timeout={args.hang_timeout}s", flush=True)

            if args.recursive:
                rel_path = adv_file.relative_to(input_path)
                base_name = adv_file.stem.replace('_adv', '')
                output_file_pt = output_path / rel_path.parent / f"{base_name}_purified.pt"
                output_file_png = output_path / rel_path.parent / f"{base_name}_purified.png"
                output_file_pt.parent.mkdir(parents=True, exist_ok=True)
            else:
                base_name = adv_file.stem.replace('_adv', '')
                output_file_pt = output_path / f"{base_name}_purified.pt"
                output_file_png = output_path / f"{base_name}_purified.png"

            if not args.no_pt:
                save_dict = {
                    'purified': x_purified.cpu(),
                    'adv': x_adv.cpu(),
                }
                if x_clean is not None:
                    save_dict['clean'] = x_clean
                torch.save(save_dict, output_file_pt)

            from torchvision.utils import save_image
            save_image(x_purified.cpu(), output_file_png)

            success_count += 1

            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (total_files - idx) * avg_time

            print(f"  [{idx}/{total_files}] {adv_file.name} - {iter_time:.1f}s (avg {avg_time:.1f}s/img, {remaining/60:.1f}min remaining)")

            if idx % 50 == 0 or idx == total_files:
                print(f"  >> Checkpoint: {idx}/{total_files} ({idx/total_files*100:.1f}%), ok={success_count}, err={error_count}")

            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        except Exception as e:
            error_count += 1
            print(f"  ERROR: {adv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - start_time
    print(f"\n")
    print("="*60)
    print(f"Done! Total time: {total_time/60:.1f} min")
    print(f"  Success: {success_count}/{total_files} ({success_count/total_files*100:.1f}%)")
    if error_count > 0:
        print(f"  Failed:  {error_count}/{total_files} ({error_count/total_files*100:.1f}%)")
    print(f"  Avg speed: {total_time/total_files:.1f}s/img")
    print("="*60)


def main():
    args = parse_args()

    print("Config:")
    print(f"  diffusion model: {args.diffusion_model_path}")
    print(f"  purification steps t: {args.t}")
    print(f"  device: {args.device}")

    print("\nLoading DiffPure purifier...")
    purifier, config = build_purifier(args)

    input_dirs = [d.strip() for d in args.input_dirs.split(',')]

    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"\nWarning: directory not found, skipping: {input_dir}")
            continue

        rel_path = input_path.relative_to(input_path.anchor)
        output_dir = Path(args.output_root) / rel_path.name

        process_directory(input_dir, output_dir, purifier, args, config)

    print(f"\nDone. Purified samples saved to: {args.output_root}")


if __name__ == '__main__':
    main()
