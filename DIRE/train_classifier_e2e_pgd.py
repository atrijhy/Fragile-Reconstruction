#!/usr/bin/env python3
"""
End-to-End PGD Adversarial Training for DIRE Classifier.
Each PGD step computes full DIRE transformation - SLOW but correct.

Supports multi-GPU training via torch.multiprocessing.spawn.
"""

import os
import sys
import argparse
import time
import glob
import random
from pathlib import Path
from datetime import datetime
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Simple config class to avoid importing utils.config which has module-level argparse
class SimpleConfig:
    def __init__(self):
        pass


def create_config(args, rank=0, world_size=1):
    cfg = SimpleConfig()
    cfg.arch = 'resnet50'
    cfg.isTrain = True
    cfg.continue_train = args.continue_train
    cfg.pretrained = True
    cfg.init_gain = 0.02
    cfg.optim = 'adam'
    cfg.lr = args.lr
    cfg.beta1 = 0.9
    cfg.nepoch = args.nepoch
    cfg.warmup = False
    cfg.new_optim = True

    # Only create exp dir on rank 0
    if rank == 0:
        exp_name = f"cls_{args.gen}_t{args.dire_t}_e2e_pgd_eps{args.e2e_eps}_k{args.e2e_k}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        exp_base = args.exp_dir or os.environ.get("DIRE_EXP_DIR", "./DIRE/data/exp")
        cfg.exp_dir = os.path.join(exp_base, exp_name)
        cfg.ckpt_dir = os.path.join(cfg.exp_dir, 'ckpt')
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
    else:
        cfg.exp_dir = None  # Will be broadcast from rank 0
        cfg.ckpt_dir = None
    cfg.epoch = args.epoch if args.continue_train else -1

    cfg.dire_mode = True
    cfg.dire_scale_to_m11 = True
    cfg.dire_apply_imagenet_norm = False
    cfg.aug_norm = False

    cfg.e2e_enable = True
    cfg.e2e_eps = args.e2e_eps / 255.0
    cfg.e2e_alpha = args.e2e_alpha / 255.0
    cfg.e2e_k = args.e2e_k
    cfg.e2e_clean_weight = args.e2e_clean_weight
    cfg.e2e_adv_weight = args.e2e_adv_weight
    cfg.e2e_adv_prob = args.e2e_adv_prob
    cfg.e2e_warmup_steps = args.e2e_warmup_steps
    cfg.e2e_log_every = args.log_every

    cfg.dire_root = args.dire_root or os.environ.get(
        "PROJECT_ROOT", str(Path(__file__).parent.parent)
    )
    cfg.dire_t_steps = args.dire_t
    cfg.dire_ode_method = args.dire_ode_method
    cfg.dire_ode_step_size = args.dire_ode_step_size
    cfg.dire_fix_rand = True
    cfg.diffusion_config_path = args.diffusion_config_path or os.environ.get(
        "DIFFPURE_CONFIG",
        str(Path(__file__).parent.parent / "DiffPure" / "configs" / "imagenet.yml"),
    )
    cfg.diffusion_model_path = args.diffusion_model_path or os.environ.get(
        "DIFFUSION_CKPT", "/path/to/256x256_diffusion_uncond.pt"
    )
    cfg.freelb_enable = False

    # Multi-GPU settings
    cfg.rank = rank
    cfg.world_size = world_size

    return cfg


class RawImageDataset(Dataset):
    """Dataset that loads raw PNG images for end-to-end adversarial training."""
    def __init__(self, real_dir, fake_dir, num_samples=None, is_train=True, aug_flip=True, max_retries=20):
        self.is_train = is_train
        self.aug_flip = aug_flip and is_train
        self.max_retries = int(max_retries)
        # Find raw PNG images (handle both .png and .PNG)
        real_candidates = sorted(
            glob.glob(os.path.join(real_dir, '**/*.png'), recursive=True) +
            glob.glob(os.path.join(real_dir, '**/*.PNG'), recursive=True)
        )
        fake_candidates = sorted(
            glob.glob(os.path.join(fake_dir, '**/*.png'), recursive=True) +
            glob.glob(os.path.join(fake_dir, '**/*.PNG'), recursive=True)
        )
        # Filter out empty/corrupted files (size > 0)
        self.real_files = [f for f in real_candidates if os.path.getsize(f) > 0]
        self.fake_files = [f for f in fake_candidates if os.path.getsize(f) > 0]
        n_filtered = (len(real_candidates) - len(self.real_files)) + (len(fake_candidates) - len(self.fake_files))
        if n_filtered > 0:
            print(f"[RawImageDataset] WARNING: Filtered out {n_filtered} empty/corrupted files")

        if num_samples:
            half = num_samples // 2
            self.real_files = self.real_files[:half]
            self.fake_files = self.fake_files[:half]
        self.samples = [(f, 0) for f in self.real_files] + [(f, 1) for f in self.fake_files]
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),  # [0, 1]
        ])
        print(f"[RawImageDataset] real={len(self.real_files)}, fake={len(self.fake_files)}, total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # NOTE: In DDP, a single rank hanging in __getitem__ will desync collectives and cause NCCL timeouts.
        # Make loading bounded and non-recursive.
        last_err = None
        start_idx = idx
        for attempt in range(self.max_retries + 1):
            img_path, label = self.samples[idx]
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    raw_image = self.transform(img)  # [3, 256, 256] in [0, 1]
                break
            except Exception as e:
                last_err = e
                # linear probe among subsequent samples to avoid infinite recursion
                idx = (idx + 1) % len(self.samples)
                continue
        else:
            # Hard fallback to keep the training step progressing.
            raw_image = torch.zeros(3, 256, 256, dtype=torch.float32)
            label = 0
            img_path = f"__fallback__:{start_idx}"
            if self.is_train:
                # Avoid spamming at eval time.
                print(f"[RawImageDataset] ERROR: failed to load after {self.max_retries} retries (idx={start_idx}). "
                      f"Last error: {last_err}")

        if self.aug_flip and torch.rand(1).item() > 0.5:
            raw_image = TF.hflip(raw_image)

        # Return raw image - DIRE will be computed on-the-fly during training
        return raw_image, label, {'source': img_path}


def _ddp_heartbeat(rank: int, batch_idx: int, every: int, msg: str) -> None:
    if every <= 0:
        return
    if batch_idx % every == 0:
        print(f"[Rank {rank}] step={batch_idx} {msg}", flush=True)


def train_epoch(trainer, dataloader, epoch, args, rank=0):
    trainer.model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Train",
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    else:
        pbar = dataloader

    data_time_avg = 0.0
    step_time_avg = 0.0
    t_data = time.time()
    for batch_idx, (raw_images, labels, meta) in enumerate(pbar):
        t_step0 = time.time()
        data_time = t_step0 - t_data
        # raw_images: [B, 3, 256, 256] in [0, 1]
        trainer.set_input((raw_images, labels, meta))
        trainer.optimize_parameters()
        t_step1 = time.time()
        step_time = t_step1 - t_step0
        t_data = t_step1

        # Exponential moving averages to keep log stable
        if batch_idx == 0:
            data_time_avg = data_time
            step_time_avg = step_time
        else:
            data_time_avg = 0.95 * data_time_avg + 0.05 * data_time
            step_time_avg = 0.95 * step_time_avg + 0.05 * step_time

        trainer.total_steps += 1
        total_loss += trainer.loss

        # Print per-step loss (rank 0 only) so logs contain every iteration's loss.
        if rank == 0:
            try:
                print(
                    f"[Train] epoch={epoch} step={trainer.total_steps} batch={batch_idx} "
                    f"loss={float(trainer.loss):.6f}",
                    flush=True,
                )
            except Exception:
                pass
        with torch.no_grad():
            if hasattr(trainer, 'output') and trainer.output is not None:
                preds = (torch.sigmoid(trainer.output.squeeze()) > 0.5).long()
                correct += (preds == trainer.label.long()).sum().item()
            total += labels.size(0)

        # Lightweight heartbeat for non-rank0 to help diagnose stalls
        if rank != 0:
            _ddp_heartbeat(rank, batch_idx, max(200, args.log_every * 10),
                           f"epoch={epoch} data_t~{data_time_avg:.3f}s step_t~{step_time_avg:.3f}s")

        # Update progress bar (only on rank 0)
        if rank == 0:
            acc = 100 * correct / total if total > 0 else 0
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc:.2f}%',
                'lr': f'{trainer.optimizer.param_groups[0]["lr"]:.1e}'
            })

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def validate(trainer, dataloader, device, rank=0):
    trainer.model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc="Validation",
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    else:
        pbar = dataloader

    with torch.no_grad():
        for batch_idx, (raw_images, labels, meta) in enumerate(pbar):
            # For validation, compute DIRE without adversarial perturbation
            trainer.set_input((raw_images, labels, meta))
            trainer.forward()
            loss = trainer.get_loss()
            total_loss += loss.item()
            preds = (torch.sigmoid(trainer.output.squeeze()) > 0.5).long()
            correct += (preds == trainer.label.long()).sum().item()
            total += labels.size(0)

            # Update progress bar (only on rank 0)
            if rank == 0:
                acc = 100 * correct / total if total > 0 else 0
                pbar.set_postfix({
                    'loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'acc': f'{acc:.2f}%'
                })

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def setup_distributed(rank, world_size, port=12355):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all GPUs."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train_worker(rank, world_size, gpu_ids, args):
    """Training worker for each GPU."""
    # Map rank to actual GPU id
    gpu_id = gpu_ids[rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    # Set seed for this worker (add rank to ensure different augmentations per GPU)
    worker_seed = args.seed
    set_seed(worker_seed)

    # Setup distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.dist_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    is_main = (rank == 0)

    # Save original argv and clear it before importing trainer
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    from utils.trainer import TrainerE2E
    sys.argv = original_argv

    # Create config - only rank 0 creates directories
    cfg = create_config(args, rank=rank, world_size=world_size)

    # Broadcast exp_dir from rank 0 to all ranks
    if is_main:
        exp_dir_tensor = torch.tensor([ord(c) for c in cfg.exp_dir] + [0] * (512 - len(cfg.exp_dir)), dtype=torch.int64, device=device)
    else:
        exp_dir_tensor = torch.zeros(512, dtype=torch.int64, device=device)
    dist.broadcast(exp_dir_tensor, src=0)

    if not is_main:
        exp_dir_chars = exp_dir_tensor.cpu().tolist()
        cfg.exp_dir = ''.join(chr(c) for c in exp_dir_chars if c != 0)
        cfg.ckpt_dir = os.path.join(cfg.exp_dir, 'ckpt')

    if is_main:
        print(f"Using {world_size} GPUs: {gpu_ids}")
        print(f"\nExperiment: {cfg.exp_dir}")
        print(f"E2E PGD: eps={args.e2e_eps}/255, alpha={args.e2e_alpha}/255, k={args.e2e_k}")

    # Override device in config
    cfg.device = device

    if is_main:
        print(f"\n[Rank {rank}] Initializing TrainerE2E on GPU {gpu_id}...")

    trainer = TrainerE2E(cfg)
    trainer.device = device
    trainer.model = trainer.model.to(device)

    # Wrap classifier in DDP (not the DIRE model - it's separate per GPU)
    trainer.model = DDP(trainer.model, device_ids=[gpu_id], find_unused_parameters=True)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        trainer.optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # Warm-up learning rate scheduler
    if cfg.warmup:
        warmup_steps = cfg.e2e_warmup_steps
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps)
        )

    # Create datasets
    if is_main:
        print("\nLoading datasets...")

    train_dataset = RawImageDataset(
        real_dir=os.path.join(args.data_root, 'real', 'train'),
        fake_dir=os.path.join(args.data_root, args.gen, 'train'),
        num_samples=args.num_samples, is_train=True, aug_flip=True, max_retries=args.max_image_retries)
    val_dataset = RawImageDataset(
        real_dir=os.path.join(args.data_root, 'real', 'val'),
        fake_dir=os.path.join(args.data_root, args.gen, 'val'),
        num_samples=args.num_val_samples, is_train=False, aug_flip=False, max_retries=args.max_image_retries)

    # Use DistributedSampler for training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.num_workers,
        pin_memory=True)

    if is_main:
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        print(f"Batches per GPU per epoch: {len(train_loader)} train, {len(val_loader)} val")

    best_acc = 0.0
    patience_counter = 0
    patience = args.patience

    for epoch in range(1, args.nepoch + 1):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        val_sampler.set_epoch(epoch)

        if is_main:
            print(f"\n{'='*60}\nEpoch {epoch}/{args.nepoch} | LR: {trainer.optimizer.param_groups[0]['lr']:.2e}\n{'='*60}")

        epoch_start = time.time()
        train_loss, train_acc = train_epoch(trainer, train_loader, epoch, args, rank=rank)

        # Aggregate metrics across GPUs.
        try:
            train_loss_tensor = torch.tensor([train_loss], device=device)
            train_acc_tensor = torch.tensor([train_acc], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / world_size
            train_acc = train_acc_tensor.item() / world_size
        except Exception as e:
            print(f"[Rank {rank}] ERROR during train metric all_reduce at epoch={epoch}: {e}", flush=True)
            raise

        train_time = time.time() - epoch_start

        if is_main:
            print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}%, Time: {train_time/60:.1f} min")

        val_loss, val_acc = validate(trainer, val_loader, device, rank=rank)

        # Aggregate validation metrics
        try:
            val_loss_tensor = torch.tensor([val_loss], device=device)
            val_acc_tensor = torch.tensor([val_acc], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / world_size
            val_acc = val_acc_tensor.item() / world_size
        except Exception as e:
            print(f"[Rank {rank}] ERROR during val metric all_reduce at epoch={epoch}: {e}", flush=True)
            raise

        if is_main:
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {100*val_acc:.2f}%")

        scheduler.step(val_acc)

        # Update warm-up learning rate
        if cfg.warmup:
            warmup_scheduler.step()

        if is_main:
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                # Save without DDP wrapper
                save_dict = {
                    'model': trainer.model.module.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                    'total_steps': trainer.total_steps,
                }
                torch.save(save_dict, os.path.join(cfg.ckpt_dir, 'model_epoch_best.pth'))
                print(f"  -> New best model! Acc: {100*best_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

            if epoch % args.save_freq == 0:
                save_dict = {
                    'model': trainer.model.module.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                    'total_steps': trainer.total_steps,
                }
                torch.save(save_dict, os.path.join(cfg.ckpt_dir, f'model_epoch_{epoch}.pth'))

            # Save latest
            save_dict = {
                'model': trainer.model.module.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'total_steps': trainer.total_steps,
            }
            torch.save(save_dict, os.path.join(cfg.ckpt_dir, 'model_epoch_latest.pth'))

        # Broadcast patience_counter to all ranks for early stopping
        patience_tensor = torch.tensor([patience_counter], device=device)
        dist.broadcast(patience_tensor, src=0)
        patience_counter = patience_tensor.item()

        if patience_counter >= patience:
            if is_main:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
            break

    # Prevent ranks from drifting too far.
    dist.barrier()

    if is_main:
        print(f"\nTraining complete! Best accuracy: {100*best_acc:.2f}%")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='E2E PGD Adversarial Training (Multi-GPU)')
    parser.add_argument('--gen', type=str, default='adm')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing real/ and <gen>/ subdirectories')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Base directory for experiment outputs (overrides DIRE_EXP_DIR env var)')
    parser.add_argument('--dire_root', type=str, default=None,
                        help='Project root directory (overrides PROJECT_ROOT env var)')
    parser.add_argument('--diffusion_config_path', type=str, default=None,
                        help='Path to DiffPure imagenet.yml config (overrides DIFFPURE_CONFIG env var)')
    parser.add_argument('--diffusion_model_path', type=str, default=None,
                        help='Path to 256x256_diffusion_uncond.pt checkpoint (overrides DIFFUSION_CKPT env var)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nepoch', type=int, default=30)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_val_samples', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--e2e_eps', type=float, default=8)
    parser.add_argument('--e2e_alpha', type=float, default=2)
    parser.add_argument('--e2e_k', type=int, default=8)
    parser.add_argument('--e2e_clean_weight', type=float, default=0.0)
    parser.add_argument('--e2e_adv_weight', type=float, default=1.0)
    parser.add_argument('--e2e_adv_prob', type=float, default=1.0)
    parser.add_argument('--e2e_warmup_steps', type=int, default=500)
    parser.add_argument('--dire_t', type=int, default=20)
    parser.add_argument('--dire_ode_method', type=str, default='euler')
    parser.add_argument('--dire_ode_step_size', type=float, default=0.05)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--epoch', type=str, default='latest')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--gpus', type=str, default='0,1,2,4,5', help='Comma-separated GPU IDs')
    parser.add_argument('--dist_port', type=int, default=12355, help='Port for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_image_retries', type=int, default=20,
                        help='Max retries in dataset __getitem__ before falling back to a dummy image')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    world_size = len(gpu_ids)

    print(f"Starting multi-GPU training on GPUs: {gpu_ids}")
    print(f"World size: {world_size}")
    print(f"Effective batch size: {args.batch_size} x {world_size} = {args.batch_size * world_size}")

    # Spawn workers
    mp.spawn(
        train_worker,
        args=(world_size, gpu_ids, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
