import os
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.nn import init
import random as pyrandom
import torch.nn.functional as F
from typing import Optional

from utils.config import CONFIGCLASS
from utils.utils import get_network
from utils.warmup import GradualWarmupScheduler

# Optional external attacker (lightweight). If not installed, we fall back to the internal PGD.
try:
    import torchattacks  # type: ignore
except Exception:
    torchattacks = None


class ModelWithMeta(nn.Module):
    """Thin wrapper that binds a `meta` object for models that accept (x, meta).

    torchattacks expects a model that takes only inputs (x). This wrapper stores
    `cur_meta` and forwards calls to the base model as base(x, cur_meta).
    """

    def __init__(self, base_model: nn.Module, preprocess_fn: Optional[Callable] = None):
        super().__init__()
        self.base = base_model
        self.cur_meta = None
        # preprocess_fn maps raw pixel input [0,1] -> model input (e.g. scale_to_m11 + normalize)
        self.preprocess_fn = preprocess_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x in raw pixel space [0,1]. Apply preprocess_fn if provided so
        # attackers can operate in pixel space while model receives preprocessed input.
        if self.preprocess_fn is not None:
            xp = self.preprocess_fn(x)
        else:
            xp = x
        out = self.base(xp, self.cur_meta)
        # torchattacks uses CrossEntropyLoss which needs [batch, n_classes>=2].
        # Our model outputs a single logit z with shape [batch,1] for BCE.
        # Convert to 2-class logits [0, z] so that softmax([0,z])[1] = sigmoid(z).
        if out.dim() == 2 and out.shape[1] == 1:
            out = torch.cat([torch.zeros_like(out), out], dim=1)
        return out

    # expose parameters for convenience
    def parameters(self, *args, **kwargs):
        return self.base.parameters(*args, **kwargs)


# ============================================
# Preprocessing functions for input-space PGD
# ============================================
# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_preprocess_fn(cfg: CONFIGCLASS) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a preprocessing function based on config.

    When pgd_input_space=True (input-space PGD):
      - Data is loaded in [0, 1] range (dire_raw_input=True)
      - PGD operates in [0, 1] space
      - Preprocessing is applied AFTER adding perturbation, BEFORE model forward

    Preprocessing order:
      1. scale_to_m11: [0, 1] -> [-1, 1]
      2. imagenet_norm: apply ImageNet mean/std normalization

    Returns a function: x ∈ [0, 1] -> preprocessed x
    """
    do_scale_m11 = bool(getattr(cfg, "pgd_preprocess_scale_to_m11", False))
    do_imagenet_norm = bool(getattr(cfg, "pgd_preprocess_imagenet_norm", False))

    if not do_scale_m11 and not do_imagenet_norm:
        # Identity preprocessing
        return lambda x: x

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        # x is in [0, 1]
        if do_scale_m11:
            x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        if do_imagenet_norm:
            mean = IMAGENET_MEAN.to(x.device, x.dtype)
            std = IMAGENET_STD.to(x.device, x.dtype)
            # Note: if scale_to_m11 was applied, we assume imagenet_norm
            # should work on the [-1, 1] or [0, 1] space as intended
            # Typically these are mutually exclusive
            x = (x - mean) / std
        return x

    return preprocess


class BaseModel(nn.Module):
    def __init__(self, cfg: CONFIGCLASS):
        super().__init__()
        self.cfg = cfg
        self.total_steps = 0
        self.isTrain = cfg.isTrain
        self.save_dir = cfg.ckpt_dir
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer

    def save_networks(self, epoch: int):
        save_filename = f"model_epoch_{epoch}.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)

    # load models from the disk
    def load_networks(self, epoch: int):
        load_filename = f"model_epoch_{epoch}.pth"
        load_path = os.path.join(self.save_dir, load_filename)

        print(f"loading the model from {load_path}")
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        # Always load checkpoints onto CPU first to avoid transient GPU OOM during torch.load().
        # The model will be moved to the training device later, and optimizer state (when used)
        # is explicitly moved below.
        state_dict = torch.load(load_path, map_location="cpu")
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        self.model.load_state_dict(state_dict["model"])
        self.total_steps = state_dict["total_steps"]

        if self.isTrain and not self.cfg.new_optim:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            # move optimizer state to GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.lr

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net: nn.Module, init_type="normal", gain=0.02):
    def init_func(m: nn.Module):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, cfg: CONFIGCLASS):
        super().__init__(cfg)
        self.arch = cfg.arch
        self.model = get_network(self.arch, cfg.isTrain, cfg.continue_train, cfg.init_gain, cfg.pretrained)

        # Input-space PGD: preprocess after perturbation
        self.pgd_input_space = bool(getattr(cfg, "pgd_input_space", False))
        if self.pgd_input_space:
            self.preprocess_fn = build_preprocess_fn(cfg)
            # Tell model to use x directly (skip loading DIRE maps from disk/cache).
            # PGD perturbs x, so the model must use the perturbed tensor.
            if hasattr(self.model, 'bypass_dire_loading'):
                self.model.bypass_dire_loading = True
        else:
            self.preprocess_fn = lambda x: x

        # Torchattacks integration flags (optional)
        self.use_torchattacks = bool(getattr(cfg, "use_torchattacks", False))
        # PGD-attack hyperparameters for torchattacks (defaults match common PGD-8)
        self.ta_eps = float(getattr(cfg, "ta_eps", 8.0 / 255.0) or 8.0 / 255.0)
        self.ta_alpha = float(getattr(cfg, "ta_alpha", 2.0 / 255.0) or 2.0 / 255.0)
        self.ta_steps = int(getattr(cfg, "ta_steps", 8) or 8)
        # Lazy-initialized wrapper and attack object
        self._ta_wrapper: Optional[ModelWithMeta] = None
        self._ta_attack = None

        self.loss_fn = nn.BCEWithLogitsLoss()
        # initialize optimizers
        if cfg.optim == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        elif cfg.optim == "sgd":
            momentum = float(getattr(cfg, "sgd_momentum", 0.9) or 0.0)
            weight_decay = float(getattr(cfg, "sgd_weight_decay", 5e-4) or 0.0)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError("optim should be [adam, sgd]")
        if cfg.warmup:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, cfg.nepoch - cfg.warmup_epoch, eta_min=1e-6
            )
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, multiplier=1, total_epoch=cfg.warmup_epoch, after_scheduler=scheduler_cosine
            )
            self.scheduler.step()
        if cfg.continue_train:
            self.load_networks(cfg.epoch)
        self.model.to(self.device)
        # When fine-tuning from an existing checkpoint, cfg.total_steps is restored from
        # that checkpoint. For FreeLB warmup we want "steps since this run started",
        # not absolute historical steps, so we keep a per-run start marker.
        self._freelb_start_total_steps: Optional[int] = None

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def set_input(self, input):
        img, label, meta = input if len(input) == 3 else (input[0], input[1], {})
        self.input = img.to(self.device)
        self.label = label.to(self.device).float()
        for k in meta.keys():
            if isinstance(meta[k], torch.Tensor):
                meta[k] = meta[k].to(self.device)
        self.meta = meta

    def forward(self):
        # Apply preprocessing when pgd_input_space=True (data in [0,1], preprocess before model)
        x = self.preprocess_fn(self.input) if self.pgd_input_space else self.input
        self.output = self.model(x, self.meta)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def _project_l2(self, delta: torch.Tensor, eps: float) -> torch.Tensor:
        # Per-sample L2 projection to radius eps.
        if eps <= 0:
            return delta.detach()
        b = delta.shape[0]
        flat = delta.view(b, -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        factors = torch.clamp(eps / norms, max=1.0)
        flat = flat * factors
        return flat.view_as(delta).detach()

    def _freelb_delta_init(self, x: torch.Tensor, eps: float, norm: str, rand_init: bool) -> torch.Tensor:
        if (not rand_init) or eps <= 0:
            return torch.zeros_like(x)
        if norm == "l2":
            delta = torch.randn_like(x)
            delta = self._project_l2(delta, eps=eps)
            # random radius in [0, eps]
            b = delta.shape[0]
            r = torch.rand(b, device=delta.device).view(b, *([1] * (delta.dim() - 1)))
            return (delta * r).detach()
        # default: linf
        return (torch.empty_like(x).uniform_(-eps, eps)).detach()

    def _optimize_parameters_pgd(self):
        """
        Standard PGD Adversarial Training (Madry et al.).

        Unlike FreeLB which accumulates gradients during the inner loop,
        PGD-AT first finds the adversarial example using K steps, then
        performs a single model update using the final adversarial loss.

        Pipeline:
        1. Initialize delta randomly in [-eps, eps]
        2. For k steps: compute loss(x+delta), update delta via gradient ascent
        3. Final: compute loss(x+delta_final), backprop, update model

        When pgd_input_space=True:
          - Data is in [0, 1] range (loaded with dire_raw_input=True)
          - PGD operates in [0, 1] space, always clamp to [0, 1]
          - Apply preprocess_fn AFTER perturbation, BEFORE model forward
          - eps/alpha are in [0, 1] scale (e.g., 8/255)
        """
        k = int(getattr(self.cfg, "pgd_k", 8) or 0)
        if k < 1:
            # fallback to standard training
            self.forward()
            loss = self.loss_fn(self.output.squeeze(1), self.label)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.loss = float(loss.detach().item())
            return

        eps = float(getattr(self.cfg, "pgd_eps", 0.0) or 0.0)
        alpha = float(getattr(self.cfg, "pgd_alpha", 0.0) or 0.0)
        norm = str(getattr(self.cfg, "pgd_norm", "linf") or "linf").strip().lower()
        rand_init = bool(getattr(self.cfg, "pgd_rand_init", True))

        if norm not in {"linf", "l2"}:
            raise ValueError(f"Unsupported pgd_norm: {norm} (expected linf|l2)")

        x = self.input
        y = self.label
        meta = self.meta

        # Input-space PGD mode: data in [0,1], preprocess after perturbation
        if self.pgd_input_space:
            preprocess = self.preprocess_fn
            # Support pgd_disable_clamp in input-space mode too
            if bool(getattr(self.cfg, "pgd_disable_clamp", False)):
                clamp_lo, clamp_hi = None, None
            else:
                clamp_lo, clamp_hi = 0.0, 1.0
        else:
            # Legacy mode: determine clamping based on dataloader preprocessing
            preprocess = lambda t: t  # no extra preprocessing
            if bool(getattr(self.cfg, "pgd_disable_clamp", False)):
                clamp_lo, clamp_hi = None, None
            elif bool(getattr(self.cfg, "dire_mode", False)):
                if bool(getattr(self.cfg, "dire_apply_imagenet_norm", False)):
                    clamp_lo, clamp_hi = None, None
                elif bool(getattr(self.cfg, "dire_scale_to_m11", False)):
                    clamp_lo, clamp_hi = -1.0, 1.0
                else:
                    clamp_lo, clamp_hi = 0.0, 1.0
            else:
                if bool(getattr(self.cfg, "aug_norm", True)):
                    clamp_lo, clamp_hi = None, None
                else:
                    clamp_lo, clamp_hi = 0.0, 1.0

        # Initialize delta
        delta = self._freelb_delta_init(x, eps=eps, norm=norm, rand_init=rand_init)

        # Optional: use torchattacks to generate adversarial examples (simpler API)
        if getattr(self, 'use_torchattacks', False):
            if torchattacks is None:
                print("[WARN] cfg.use_torchattacks=True but torchattacks is not installed; falling back to native PGD implementation.")
            else:
                # initialize wrapper and attack lazily
                if self._ta_wrapper is None:
                    # if model is DataParallel/DistributedDataParallel, use .module for the base
                    base = self.model.module if hasattr(self.model, 'module') else self.model
                    self._ta_wrapper = ModelWithMeta(base)
                if self._ta_attack is None:
                    # instantiate torchattacks.PGD with wrapper
                    self._ta_attack = torchattacks.PGD(self._ta_wrapper, eps=self.ta_eps, alpha=self.ta_alpha, steps=self.ta_steps)

                # Prepare inputs for attack
                # wrapper expects to be on same device
                self._ta_wrapper.to(x.device)
                self._ta_wrapper.cur_meta = meta

                # Ensure labels are long for classification if needed (BCE with logits expects floats but attackers often use class labels)
                labels_for_attack = y.long() if y.dtype == torch.float32 or y.dtype == torch.float64 else y

                # generate adv images (temporarily set model to eval to avoid BN updates)
                self.model.eval()
                with torch.no_grad():
                    try:
                        adv_images = self._ta_attack(x, labels_for_attack)
                    except Exception:
                        # In case attacker fails (signature mismatch), fall back to native implementation
                        adv_images = None
                self.model.train()

                if adv_images is not None:
                    # Use adv_images as the adversarial batch and perform a single training update
                    # Preprocess adv_images if needed (when pgd_input_space True preprocess will be applied in forward)
                    x_adv_final = adv_images
                    if clamp_lo is not None:
                        x_adv_final = x_adv_final.clamp(clamp_lo, clamp_hi)

                    x_adv_final = preprocess(x_adv_final) if self.pgd_input_space else x_adv_final
                    out_final = self.model(x_adv_final, meta)
                    adv_loss = self.loss_fn(out_final.squeeze(1), y)
                    # Optionally include clean loss as well
                    clean_weight = float(getattr(self.cfg, "pgd_clean_weight", 0.0) or 0.0)
                    if clean_weight > 0:
                        x_clean = preprocess(x) if self.pgd_input_space else x
                        out_clean = self.model(x_clean, meta)
                        clean_loss = self.loss_fn(out_clean.squeeze(1), y)
                        total_loss = clean_weight * clean_loss + adv_loss
                    else:
                        total_loss = adv_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    self.optimizer.step()
                    self.output = out_final.detach()
                    self.loss = float(total_loss.detach().item())
                    return

        # PGD inner loop: find adversarial perturbation (NO model gradient accumulation)
        for step in range(k):
            delta.requires_grad_(True)
            x_adv = x + delta
            if clamp_lo is not None:
                x_adv = x_adv.clamp(clamp_lo, clamp_hi)

            # Apply preprocessing (only effective when pgd_input_space=True)
            x_adv_preprocessed = preprocess(x_adv)

            # Forward pass to compute delta gradient (model params stay frozen in inner loop)
            out = self.model(x_adv_preprocessed, meta)
            loss = self.loss_fn(out.squeeze(1), y)

            # Compute gradient w.r.t. delta only
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

            # Update delta (gradient ascent to maximize loss) - wrap in no_grad for cleanliness
            with torch.no_grad():
                if norm == "l2":
                    b = grad.shape[0]
                    g = grad.view(b, -1)
                    g_norm = g.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    g = g / g_norm
                    d = delta.view(b, -1) + float(alpha) * g
                    delta = self._project_l2(d.view_as(delta), eps=eps).detach()
                else:
                    # Linf: use sign of gradient
                    delta = (delta + alpha * grad.sign()).clamp(-eps, eps).detach()

        # Final adversarial forward pass for training
        x_adv_final = x + delta.detach()
        if clamp_lo is not None:
            x_adv_final = x_adv_final.clamp(clamp_lo, clamp_hi)
        x_adv_final = preprocess(x_adv_final)

        # Get weights for clean/adv loss mix
        clean_weight = float(getattr(self.cfg, "pgd_clean_weight", 0.0) or 0.0)
        adv_weight = float(getattr(self.cfg, "pgd_adv_weight", 1.0) or 1.0)

        # Single model update with mixed loss (clean + adversarial)
        self.optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        
        # Clean loss (on original input)
        # NOTE: even if clean_weight==0 we may still need out_clean for consistency/KL losses.
        out_clean = None
        clean_loss = None
        adv_loss_type = str(getattr(self.cfg, "pgd_adv_loss", "bce") or "bce").strip().lower()
        need_clean_out = (clean_weight > 0) or (adv_loss_type in {"kl", "kl_div", "consistency_kl"})
        if need_clean_out:
            # Apply preprocessing for clean input too (when pgd_input_space=True)
            x_clean = preprocess(x)
            out_clean = self.model(x_clean, meta)
        if clean_weight > 0:
            clean_loss = self.loss_fn(out_clean.squeeze(1), y)
            total_loss = total_loss + clean_weight * clean_loss

        # Adversarial term (x_adv_final is already preprocessed above)
        out_final = self.model(x_adv_final, meta)

        if adv_loss_type in {"bce", "ce", "cls"}:
            # Standard PGD-AT classification loss
            adv_loss = self.loss_fn(out_final.squeeze(1), y)
            total_loss = total_loss + adv_weight * adv_loss
        elif adv_loss_type in {"kl", "kl_div", "consistency_kl"}:
            # Consistency regularization: KL(p_clean || p_adv)
            # This trainer is binary (BCEWithLogitsLoss), so out_* are logits of a Bernoulli.
            # Use Bernoulli KL in probability space for numerical stability.
            if out_clean is None:
                x_clean = preprocess(x)
                out_clean = self.model(x_clean, meta)
            p = torch.sigmoid(out_clean.squeeze(1)).clamp(1e-6, 1 - 1e-6)
            q = torch.sigmoid(out_final.squeeze(1)).clamp(1e-6, 1 - 1e-6)
            adv_loss = (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()
            beta = float(getattr(self.cfg, "pgd_kl_beta", 1.0) or 1.0)
            total_loss = total_loss + (adv_weight * beta) * adv_loss
        else:
            raise ValueError(f"Unsupported pgd_adv_loss: {adv_loss_type} (expected bce|kl)")
        
        total_loss.backward()
        self.optimizer.step()

        self.output = out_final.detach()
        # Report total loss (clean + adv) for accurate logging
        self.loss = float(total_loss.detach().item())

        # Optional debug logging
        # Debug delta logging is extremely verbose; keep it off by default.
        # Enable by setting cfg.pgd_log_delta_every > 0.
        log_every = int(getattr(self.cfg, "pgd_log_delta_every", 0) or 0)
        if log_every > 0 and int(getattr(self, "total_steps", 0) or 0) % log_every == 0:
            try:
                d = delta.detach()
                mean_abs = float(d.abs().mean().item())
                max_abs = float(d.abs().amax().item())
                print(
                    f"[PGD-AT] step={int(getattr(self, 'total_steps', 0) or 0)} "
                    f"|delta| mean={mean_abs:.6f} max={max_abs:.6f} "
                    f"total_loss={total_loss.item():.4f} adv_loss={adv_loss.item():.4f}"
                )
            except Exception:
                pass

    def _optimize_parameters_freelb(self):
        """
        FreeLB-style adversarial training in input space (works for both dire_mode and non-dire mode).

        We run K inner steps where we:
          - compute loss on x+delta
          - accumulate parameter grads (loss/K)
          - update delta with gradient ascent under an eps ball
        Then take ONE optimizer step.
        """
        k = int(getattr(self.cfg, "freelb_k", 3) or 0)
        if k < 1:
            # fallback to standard training
            self.forward()
            loss = self.loss_fn(self.output.squeeze(1), self.label)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.loss = float(loss.detach().item())
            return

        eps = float(getattr(self.cfg, "freelb_eps", 0.0) or 0.0)
        alpha = float(getattr(self.cfg, "freelb_alpha", 0.0) or 0.0)
        norm = str(getattr(self.cfg, "freelb_norm", "linf") or "linf").strip().lower()
        rand_init = bool(getattr(self.cfg, "freelb_rand_init", True))
        clean_w = float(getattr(self.cfg, "freelb_clean_weight", 1.0) or 0.0)
        adv_w = float(getattr(self.cfg, "freelb_adv_weight", 1.0) or 0.0)
        adv_prob = float(getattr(self.cfg, "freelb_adv_prob", 1.0) or 0.0)
        if clean_w < 0 or adv_w < 0:
            raise ValueError("freelb_clean_weight/freelb_adv_weight must be >= 0")
        if adv_prob < 0 or adv_prob > 1:
            raise ValueError("freelb_adv_prob must be in [0,1]")
        if norm not in {"linf", "l2"}:
            raise ValueError(f"Unsupported freelb_norm: {norm} (expected linf|l2)")

        x = self.input
        y = self.label
        meta = self.meta

        # Stochastic application (after warmup handled in optimize_parameters)
        # Stochastically apply adversarial inner-loop at the batch level.
        # Use a 0-dim random tensor so `.item()` is always valid, and sample on the same device.
        if adv_w <= 0 or adv_prob <= 0 or (adv_prob < 1.0 and torch.rand((), device=x.device).item() > adv_prob):
            self.forward()
            loss = self.loss_fn(self.output.squeeze(1), self.label)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.loss = float(loss.detach().item())
            return

        total_w = clean_w + adv_w
        if total_w <= 0:
            # nothing to do; fallback to clean
            self.forward()
            loss = self.loss_fn(self.output.squeeze(1), self.label)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.loss = float(loss.detach().item())
            return

        self.optimizer.zero_grad(set_to_none=True)
        clean_loss: Optional[torch.Tensor] = None
        if clean_w > 0:
            out_clean = self.model(x, meta)
            clean_loss = self.loss_fn(out_clean.squeeze(1), y)
            (clean_loss * (clean_w / total_w)).backward()

        delta = self._freelb_delta_init(x, eps=eps, norm=norm, rand_init=rand_init)

        # Keep adversarial examples in a valid input range when possible.
        # - dire_apply_imagenet_norm / aug_norm: input is already standardized -> skip clamping.
        # - dire_scale_to_m11: input in [-1, 1] -> clamp to [-1, 1].
        # - otherwise: assume input in [0, 1] -> clamp to [0, 1].
        clamp_mode: Optional[str] = None
        if bool(getattr(self.cfg, "dire_mode", False)):
            if bool(getattr(self.cfg, "dire_apply_imagenet_norm", False)):
                clamp_mode = None
            elif bool(getattr(self.cfg, "dire_scale_to_m11", False)):
                clamp_mode = "m11"
            else:
                clamp_mode = "01"
        else:
            if bool(getattr(self.cfg, "aug_norm", True)):
                clamp_mode = None
            else:
                clamp_mode = "01"

        last_loss: Optional[torch.Tensor] = None
        for _ in range(k):
            delta.requires_grad_(True)
            x_adv = x + delta
            if clamp_mode == "m11":
                x_adv = x_adv.clamp(-1.0, 1.0)
            elif clamp_mode == "01":
                x_adv = x_adv.clamp(0.0, 1.0)

            out = self.model(x_adv, meta)
            loss = self.loss_fn(out.squeeze(1), y)
            # keep overall gradient scale comparable to non-FreeLB
            (loss * (adv_w / total_w) / float(k)).backward()
            last_loss = loss.detach()

            if eps <= 0 or alpha <= 0:
                delta = delta.detach()
                continue
            if delta.grad is None:
                raise RuntimeError("FreeLB: delta.grad is None; expected gradients on delta after backward()")
            grad = delta.grad.detach()
            # IMPORTANT: clear grad to avoid accidental accumulation / memory growth
            # if delta is reused or debugging adds extra backward passes.
            delta.grad = None
            if norm == "l2":
                b = grad.shape[0]
                g = grad.view(b, -1)
                g_norm = g.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                g = g / g_norm
                d = delta.detach().view(b, -1) + float(alpha) * g
                delta = self._project_l2(d.view_as(delta), eps=eps)
            else:
                # Linf update: normalize by per-sample ||grad||_inf (matches common FreeLB implementation),
                # then project into the Linf ball. This is slightly smoother than grad.sign().
                b = grad.shape[0]
                g = grad.view(b, -1)
                g_inf = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                g = g / g_inf
                d = delta.detach().view(b, -1) + float(alpha) * g
                delta = d.view_as(delta).clamp(-eps, eps).detach()

        # Optional debug: log perturbation magnitude to confirm adversarial training is active.
        log_every = int(getattr(self.cfg, "freelb_log_delta_every", 0) or 0)
        if log_every > 0 and int(getattr(self, "total_steps", 0) or 0) % log_every == 0:
            try:
                d = delta.detach()
                mean_abs = float(d.abs().mean().item())
                max_abs = float(d.abs().amax().item())
                ll = float(last_loss.item()) if last_loss is not None else float("nan")
                print(
                    f"[freelb] step={int(getattr(self, 'total_steps', 0) or 0)} "
                    f"|delta| mean={mean_abs:.6f} max={max_abs:.6f} last_loss={ll:.6f}"
                )
            except Exception:
                pass

        self.optimizer.step()
        if last_loss is None:
            last_loss = torch.tensor(float("nan"))
        total_loss = (clean_loss * (clean_w / total_w)) if clean_loss is not None else 0.0
        total_loss = total_loss + (last_loss * (adv_w / total_w))
        self.loss = float(total_loss.item() if torch.is_tensor(total_loss) else float(total_loss))

    def optimize_parameters(self):
        # Check for PGD-AT mode first
        if bool(getattr(self.cfg, "pgd_enable", False)):
            warmup_steps = int(getattr(self.cfg, "pgd_warmup_steps", 0) or 0)
            if self._freelb_start_total_steps is None:
                self._freelb_start_total_steps = int(getattr(self, "total_steps", 0) or 0)
            local_steps = int(getattr(self, "total_steps", 0) or 0) - int(self._freelb_start_total_steps or 0)
            if warmup_steps > 0 and local_steps < warmup_steps:
                if local_steps == 0:
                    try:
                        print(f"[PGD-AT] warmup enabled: {warmup_steps} steps")
                    except Exception:
                        pass
                self.forward()
                loss = self.loss_fn(self.output.squeeze(1), self.label)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.loss = float(loss.detach().item())
                return
            self._optimize_parameters_pgd()
            return

        if bool(getattr(self.cfg, "freelb_enable", False)):
            warmup_steps = int(getattr(self.cfg, "freelb_warmup_steps", 0) or 0)
            # Warmup is measured relative to the start of *this* run, not the absolute
            # total_steps restored from a previous checkpoint.
            if self._freelb_start_total_steps is None:
                self._freelb_start_total_steps = int(getattr(self, "total_steps", 0) or 0)
            local_steps = int(getattr(self, "total_steps", 0) or 0) - int(self._freelb_start_total_steps or 0)
            if warmup_steps > 0 and local_steps < warmup_steps:
                # Warm up with clean training to avoid early collapse.
                if local_steps == 0:
                    try:
                        print(
                            f"[freelb] warmup enabled: {warmup_steps} steps (relative); "
                            f"start_total_steps={self._freelb_start_total_steps}"
                        )
                    except Exception:
                        pass
                self.forward()
                loss = self.loss_fn(self.output.squeeze(1), self.label)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.loss = float(loss.detach().item())
                return

            self._optimize_parameters_freelb()
            return

        self.forward()
        loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.loss = float(loss.detach().item())


class TrainerE2E(Trainer):
    """
    End-to-End PGD Adversarial Trainer for DIRE.
    
    This trainer performs adversarial training where perturbations are added
    to the raw input image, then the perturbed image goes through the full
    DIRE transformation before classification.
    
    Pipeline:
        x_raw --[+delta]--> x_adv --[DIRE_ODE]--> dire_map --[classifier]--> logits
    """
    
    def name(self):
        return "TrainerE2E"
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self._e2e_start_total_steps: Optional[int] = None
        self.dire_runner = None
        # Use device from cfg if available (for multi-GPU)
        if hasattr(cfg, 'device') and cfg.device is not None:
            self.device = cfg.device
        self._init_dire_model()
    
    def forward(self):
        """
        Forward pass for E2E mode.
        Input is raw image [B, 3, 256, 256] in [0, 1].
        Compute DIRE map and then classify.
        """
        raw_image = self.input  # [B, 3, 256, 256] in [0, 1]
        dire_map = self.compute_dire_map(raw_image)
        cls_input = self.preprocess_dire_for_classifier(dire_map)
        self.output = self._forward_classifier(cls_input, self.meta)
    
    def _forward_classifier(self, x, meta):
        """Forward through classifier, handling DDP wrapper."""
        if hasattr(self.model, 'module'):
            return self.model.module(x, meta)
        else:
            return self.model(x, meta)
    
    def _init_dire_model(self):
        """Initialize DIRE ODE model for computing DIRE maps."""
        import yaml
        import sys
        import argparse as ap
        
        from runners.dire_ode_model import DIRE_ODE_Model, ODEDireConfig

        # Load diffusion config
        config_path = getattr(self.cfg, 'diffusion_config_path',
                              os.environ.get("DIFFPURE_CONFIG", "/path/to/DiffPure/configs/imagenet.yml"))
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        def dict2namespace(d):
            ns = ap.Namespace()
            for k, v in d.items():
                if isinstance(v, dict):
                    v = dict2namespace(v)
                setattr(ns, k, v)
            return ns
        
        config = dict2namespace(config)
        config.device = self.device
        
        # Create args namespace for DIRE model
        dire_args = ap.Namespace()
        dire_args.diffusion_model_path = getattr(self.cfg, 'diffusion_model_path',
            os.environ.get("DIFFUSION_CKPT", "/path/to/256x256_diffusion_uncond.pt"))
        dire_args.seed = 42
        dire_args.t = int(getattr(self.cfg, 'dire_t_steps', 20))
        dire_args.eot_reps = 1
        dire_args.fix_rand = bool(getattr(self.cfg, 'dire_fix_rand', True))
        
        # ODE config - DISABLE use_unet_checkpoint for adjoint ODE compatibility!
        ode_cfg = ODEDireConfig(
            t_steps=int(getattr(self.cfg, 'dire_t_steps', 20)),
            method=str(getattr(self.cfg, 'dire_ode_method', 'euler')),
            rtol=1e-3,
            atol=1e-3,
            step_size=float(getattr(self.cfg, 'dire_ode_step_size', 0.05)),
            fix_rand=bool(getattr(self.cfg, 'dire_fix_rand', True)),
            eot_reps=1,
            clamp_recon=True,
            dire_scale_to_01=True,
            dire_smooth_eps=1e-5,
            dire_log_grad=False,
            force_fp32=True,
            nan_fallback=True,
            use_unet_checkpoint=False,  # MUST be False for adjoint gradient flow!
            disable_solver_fallback=True,
            local_half=False,
            freeze_model_params=False,
            empty_cache_each_iter=True,
        )
        
        print(f"[TrainerE2E] Initializing DIRE ODE model with t={ode_cfg.t_steps}, method={ode_cfg.method}")
        self.dire_runner = DIRE_ODE_Model(dire_args, config, ode_cfg=ode_cfg, device=self.device)
        self.dire_runner.eval()
        
        # NOTE: Do NOT freeze DIRE model parameters! 
        # odeint_adjoint requires parameters to have requires_grad=True for backward pass.
        # The optimizer only updates classifier parameters, so DIRE weights won't change.
        
        print(f"[TrainerE2E] DIRE model loaded successfully")
    
    def compute_dire_map(self, x_01: torch.Tensor) -> torch.Tensor:
        """
        Compute DIRE map from raw image.
        
        Args:
            x_01: [B, 3, 256, 256] in [0, 1] range
            
        Returns:
            dire_map: [B, 3, 256, 256] in [0, 1] range
        """
        # Convert [0, 1] -> [-1, 1] for DIRE input
        x_m11 = (x_01 - 0.5) * 2.0
        
        with torch.no_grad():
            out = self.dire_runner.forward(
                x_m11,
                t_steps=int(getattr(self.cfg, 'dire_t_steps', 20)),
                eot_reps=1,
                fix_rand=bool(getattr(self.cfg, 'dire_fix_rand', True))
            )
            dire_map = out['dire_map']  # [B, 3, H, W] in [0, 1]
        
        return dire_map
    
    def compute_dire_map_differentiable(self, x_01: torch.Tensor) -> torch.Tensor:
        """
        Compute DIRE map with gradients flowing back to x_01.
        
        This is EXPENSIVE because it requires backprop through the ODE solver!
        Uses odeint_adjoint for memory-efficient gradients.
        
        Args:
            x_01: Input image [B, 3, 256, 256] in [0, 1], MUST have requires_grad=True
            
        Returns:
            dire_map: [B, 3, H, W] in [0, 1] with gradient connection to x_01
        """
        # Convert [0, 1] -> [-1, 1] for DIRE input (preserve gradient flow!)
        x_m11 = x_01 * 2.0 - 1.0
        
        # Forward with gradients enabled through ODE solver
        out = self.dire_runner.forward(
            x_m11,
            t_steps=int(getattr(self.cfg, 'dire_t_steps', 20)),
            eot_reps=1,
            fix_rand=bool(getattr(self.cfg, 'dire_fix_rand', True))
        )
        dire_map = out['dire_map']  # [B, 3, H, W] in [0, 1]
        
        return dire_map
    
    def compute_dire_map_and_recon_differentiable(self, x_01: torch.Tensor):
        """
        Compute DIRE map AND reconstruction with full gradient flow.
        Returns both so we can compute gradients w.r.t. input through diffusion.
        
        Args:
            x_01: Input image [B, 3, 256, 256] in [0, 1]
            
        Returns:
            (dire_map, recon): Both in [0, 1] range with gradient connection
        """
        # Convert [0, 1] -> [-1, 1] for DIRE input (preserve gradient flow!)
        x_m11 = x_01 * 2.0 - 1.0
        
        out = self.dire_runner.forward(
            x_m11,
            t_steps=int(getattr(self.cfg, 'dire_t_steps', 20)),
            eot_reps=1,
            fix_rand=bool(getattr(self.cfg, 'dire_fix_rand', True))
        )
        
        dire_map = out['dire_map']  # [B, 3, H, W] in [0, 1]
        recon = out['recon']  # [B, 3, H, W] in [-1, 1]
        recon_01 = (recon + 1.0) / 2.0  # Convert to [0, 1]
        
        return dire_map, recon_01
    
    def preprocess_dire_for_classifier(self, dire_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocess DIRE map for classifier input.
        
        Args:
            dire_map: [B, 3, 256, 256] in [0, 1]
            
        Returns:
            classifier_input: [B, 3, 224, 224] in [-1, 1]
        """
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        
        # Ensure 256x256
        if dire_map.shape[-2:] != (256, 256):
            dire_map = TF.resize(dire_map, [256, 256], 
                                interpolation=InterpolationMode.BILINEAR, antialias=True)
        
        # Center crop to 224
        dire_map = TF.center_crop(dire_map, 224)
        
        # Scale to [-1, 1]
        dire_map = dire_map * 2.0 - 1.0
        
        return dire_map
    
    def _optimize_parameters_e2e_pgd(self):
        """
        End-to-End PGD adversarial training.
        
        Pipeline:
        1. Get raw image x from meta['raw_image']
        2. PGD attack in raw image space:
           - For k steps: x_adv = x + delta, compute DIRE(x_adv), get loss, update delta
        3. Final forward with x_adv, compute DIRE, classify, backprop to update classifier
        """
        k = int(getattr(self.cfg, 'e2e_k', 8))
        eps = float(getattr(self.cfg, 'e2e_eps', 8/255))
        alpha = float(getattr(self.cfg, 'e2e_alpha', 2/255))
        clean_w = float(getattr(self.cfg, 'e2e_clean_weight', 0.0))
        adv_w = float(getattr(self.cfg, 'e2e_adv_weight', 1.0))
        adv_prob = float(getattr(self.cfg, 'e2e_adv_prob', 1.0))
        
        # Get raw image: either from meta['raw_image'] or directly from self.input
        raw_image = self.meta.get('raw_image', None)
        if raw_image is None:
            # In E2E mode, self.input IS the raw image
            raw_image = self.input
        y = self.label
        
        if raw_image is None:
            # Fallback: no image available
            raise ValueError("No raw image available for E2E PGD training")
        
        raw_image = raw_image.to(self.device)  # [B, 3, 256, 256] in [0, 1]
        
        # Stochastic application
        if adv_w <= 0 or adv_prob <= 0 or (adv_prob < 1.0 and torch.rand(()).item() > adv_prob):
            # Clean training only
            dire_map = self.compute_dire_map(raw_image)
            cls_input = self.preprocess_dire_for_classifier(dire_map)
            out = self._forward_classifier(cls_input, {})
            loss = self.loss_fn(out.squeeze(1), y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.loss = float(loss.detach().item())
            self.output = out.detach()
            return
        
        total_w = clean_w + adv_w
        if total_w <= 0:
            total_w = 1.0
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # ====== Clean loss (optional) ======
        clean_loss = None
        if clean_w > 0:
            dire_map_clean = self.compute_dire_map(raw_image)
            cls_input_clean = self.preprocess_dire_for_classifier(dire_map_clean)
            out_clean = self._forward_classifier(cls_input_clean, {})
            clean_loss = self.loss_fn(out_clean.squeeze(1), y)
            (clean_loss * (clean_w / total_w)).backward()
        
        # ====== PGD attack in raw image space ======
        # TRUE E2E: gradients flow through diffusion model via odeint_adjoint!
        # Initialize delta
        delta = torch.zeros_like(raw_image).uniform_(-eps, eps)
        delta = delta.clamp(-eps, eps)
        
        # PGD inner loop - find adversarial perturbation
        # Each step: x_adv -> DIRE(x_adv) -> classifier -> loss -> grad w.r.t. delta
        for step in range(k):
            delta = delta.detach().requires_grad_(True)
            
            # x_adv with gradient connection to delta
            x_adv = raw_image.detach() + delta  # gradient flows through delta
            x_adv = x_adv.clamp(0.0, 1.0)
            
            # ====== TRUE E2E: Compute DIRE with gradients through ODE! ======
            dire_map_adv = self.compute_dire_map_differentiable(x_adv)
            cls_input_adv = self.preprocess_dire_for_classifier(dire_map_adv)
            
            # Forward through classifier
            out_adv = self._forward_classifier(cls_input_adv, {})
            loss_adv = self.loss_fn(out_adv.squeeze(1), y)
            
            # Compute gradient w.r.t delta (through x_adv -> DIRE ODE -> classifier)
            # This is the expensive part - backprop through entire ODE solver!
            grad = torch.autograd.grad(loss_adv, delta, retain_graph=False, create_graph=False)[0]
            
            # PGD step: maximize loss
            delta = delta.detach() + alpha * grad.sign()
            delta = delta.clamp(-eps, eps)
        
        # ====== Final adversarial forward pass for training ======
        x_adv_final = (raw_image + delta).clamp(0.0, 1.0)
        
        # Compute final DIRE map (no gradients needed for DIRE, only for classifier)
        dire_map_final = self.compute_dire_map(x_adv_final)
        cls_input_final = self.preprocess_dire_for_classifier(dire_map_final)
        
        # Forward through classifier and update
        out_final = self._forward_classifier(cls_input_final, {})
        adv_loss = self.loss_fn(out_final.squeeze(1), y)
        (adv_loss * (adv_w / total_w)).backward()
        
        self.optimizer.step()
        
        # Store output for accuracy computation
        self.output = out_final.detach()
        
        # Compute total loss for logging
        total_loss = 0.0
        if clean_loss is not None:
            total_loss += clean_loss.detach().item() * (clean_w / total_w)
        total_loss += adv_loss.detach().item() * (adv_w / total_w)
        self.loss = total_loss
        
        # Optional logging
        log_every = int(getattr(self.cfg, 'e2e_log_every', 0))
        if log_every > 0 and self.total_steps % log_every == 0:
            delta_mean = delta.abs().mean().item()
            delta_max = delta.abs().max().item()
            print(f"[E2E-PGD] step={self.total_steps} |delta| mean={delta_mean:.6f} max={delta_max:.6f} "
                  f"adv_loss={adv_loss.item():.4f}")
    
    def _optimize_parameters_e2e_freelb(self):
        """
        End-to-End FreeLB adversarial training.
        
        FreeLB is more efficient than PGD because it accumulates gradients
        during the inner loop, then performs a single optimizer update.
        This avoids the separate "find adversarial example" and "train" phases.
        
        Pipeline for each inner step:
        1. x_adv = x + delta
        2. Compute DIRE(x_adv) -> dire_map
        3. Classify dire_map -> loss
        4. Accumulate classifier gradients (loss / K)
        5. Update delta via gradient ascent
        After K steps: single optimizer.step()
        """
        k = int(getattr(self.cfg, 'e2e_k', 3))  # FreeLB typically uses fewer steps
        eps = float(getattr(self.cfg, 'e2e_eps', 8/255))
        alpha = float(getattr(self.cfg, 'e2e_alpha', 2/255))
        clean_w = float(getattr(self.cfg, 'e2e_clean_weight', 0.0))
        adv_w = float(getattr(self.cfg, 'e2e_adv_weight', 1.0))
        adv_prob = float(getattr(self.cfg, 'e2e_adv_prob', 1.0))
        norm = str(getattr(self.cfg, 'e2e_norm', 'linf')).strip().lower()
        rand_init = bool(getattr(self.cfg, 'e2e_rand_init', True))
        
        # Get raw image
        raw_image = self.meta.get('raw_image', None)
        if raw_image is None:
            raw_image = self.input
        y = self.label
        
        if raw_image is None:
            raise ValueError("No raw image available for E2E FreeLB training")
        
        raw_image = raw_image.to(self.device)  # [B, 3, 256, 256] in [0, 1]
        
        # Stochastic application
        if adv_w <= 0 or adv_prob <= 0 or (adv_prob < 1.0 and torch.rand(()).item() > adv_prob):
            # Clean training only
            dire_map = self.compute_dire_map(raw_image)
            cls_input = self.preprocess_dire_for_classifier(dire_map)
            out = self._forward_classifier(cls_input, {})
            loss = self.loss_fn(out.squeeze(1), y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.loss = float(loss.detach().item())
            self.output = out.detach()
            return
        
        total_w = clean_w + adv_w
        if total_w <= 0:
            total_w = 1.0
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # ====== Clean loss (optional) ======
        clean_loss = None
        if clean_w > 0:
            dire_map_clean = self.compute_dire_map(raw_image)
            cls_input_clean = self.preprocess_dire_for_classifier(dire_map_clean)
            out_clean = self._forward_classifier(cls_input_clean, {})
            clean_loss = self.loss_fn(out_clean.squeeze(1), y)
            (clean_loss * (clean_w / total_w)).backward()
        
        # ====== FreeLB: Initialize delta ======
        if rand_init and eps > 0:
            if norm == "l2":
                delta = torch.randn_like(raw_image)
                delta = self._project_l2(delta, eps=eps)
                b = delta.shape[0]
                r = torch.rand(b, device=delta.device).view(b, *([1] * (delta.dim() - 1)))
                delta = (delta * r).detach()
            else:
                # Linf
                delta = torch.empty_like(raw_image).uniform_(-eps, eps).detach()
        else:
            delta = torch.zeros_like(raw_image)
        
        # ====== FreeLB inner loop ======
        # NOTE: This is TRUE E2E - gradients flow through the diffusion model!
        # x_raw -> x_adv -> DIRE(x_adv) -> classifier -> loss
        # Gradient path: loss -> classifier -> DIRE (via odeint_adjoint) -> x_adv -> delta
        last_loss: Optional[torch.Tensor] = None
        last_output = None
        
        for step in range(k):
            # Create x_adv with gradient connection to delta
            # raw_image is fixed, delta is the variable we're optimizing
            delta = delta.detach().requires_grad_(True)
            
            # x_adv must preserve gradient to delta for E2E backprop
            x_adv = raw_image.detach() + delta  # gradient flows through delta
            x_adv = x_adv.clamp(0.0, 1.0)  # clamp preserves gradients
            
            # ====== TRUE E2E: Compute DIRE with gradients through ODE! ======
            # This calls odeint_adjoint internally which provides gradients
            dire_map_adv = self.compute_dire_map_differentiable(x_adv)
            cls_input_adv = self.preprocess_dire_for_classifier(dire_map_adv)
            
            # Forward through classifier
            out_adv = self._forward_classifier(cls_input_adv, {})
            loss_adv = self.loss_fn(out_adv.squeeze(1), y)
            
            # Accumulate classifier gradients (averaged over K steps)
            # This is the key difference from PGD!
            (loss_adv * (adv_w / total_w) / float(k)).backward()
            
            last_loss = loss_adv.detach()
            last_output = out_adv.detach()
            
            # Update delta for next step (if not last step)
            if eps <= 0 or alpha <= 0:
                delta = delta.detach()
                continue
            
            if delta.grad is None:
                raise RuntimeError("E2E FreeLB: delta.grad is None; expected gradients on delta after backward()")
            
            grad = delta.grad.detach()
            delta.grad = None
            
            if norm == "l2":
                b = grad.shape[0]
                g = grad.view(b, -1)
                g_norm = g.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                g = g / g_norm
                d = delta.detach().view(b, -1) + float(alpha) * g
                delta = self._project_l2(d.view_as(delta), eps=eps)
            else:
                # Linf: gradient ascent to maximize loss
                b = grad.shape[0]
                g = grad.view(b, -1)
                g_inf = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                g = g / g_inf
                d = delta.detach().view(b, -1) + float(alpha) * g
                delta = d.view_as(delta).clamp(-eps, eps).detach()
        
        # ====== Single optimizer step (key efficiency gain!) ======
        self.optimizer.step()
        
        # Store output for accuracy computation
        self.output = last_output
        
        # Compute total loss for logging
        total_loss = 0.0
        if clean_loss is not None:
            total_loss += clean_loss.detach().item() * (clean_w / total_w)
        if last_loss is not None:
            total_loss += last_loss.item() * (adv_w / total_w)
        self.loss = total_loss
        
        # Optional logging
        log_every = int(getattr(self.cfg, 'e2e_log_every', 0))
        if log_every > 0 and self.total_steps % log_every == 0:
            delta_mean = delta.abs().mean().item()
            delta_max = delta.abs().max().item()
            print(f"[E2E-FreeLB] step={self.total_steps} |delta| mean={delta_mean:.6f} max={delta_max:.6f} "
                  f"adv_loss={last_loss.item() if last_loss is not None else 0:.4f}")

    def optimize_parameters(self):
        """Override to use E2E adversarial training (FreeLB or PGD)."""
        # Check for E2E FreeLB first (more efficient)
        if bool(getattr(self.cfg, 'e2e_freelb_enable', False)):
            warmup_steps = int(getattr(self.cfg, 'e2e_warmup_steps', 0))
            
            if self._e2e_start_total_steps is None:
                self._e2e_start_total_steps = self.total_steps
            
            local_steps = self.total_steps - self._e2e_start_total_steps
            
            if warmup_steps > 0 and local_steps < warmup_steps:
                if local_steps == 0:
                    print(f"[E2E-FreeLB] warmup enabled: {warmup_steps} steps")
                
                self.forward()
                loss = self.loss_fn(self.output.squeeze(1), self.label)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.loss = float(loss.detach().item())
                return
            
            self._optimize_parameters_e2e_freelb()
            return
        
        # E2E PGD (original, slower)
        if bool(getattr(self.cfg, 'e2e_enable', False)):
            warmup_steps = int(getattr(self.cfg, 'e2e_warmup_steps', 0))
            
            if self._e2e_start_total_steps is None:
                self._e2e_start_total_steps = self.total_steps
            
            local_steps = self.total_steps - self._e2e_start_total_steps
            
            if warmup_steps > 0 and local_steps < warmup_steps:
                # Warmup with clean training
                if local_steps == 0:
                    print(f"[E2E-PGD] warmup enabled: {warmup_steps} steps")
                
                # Use pre-computed dire_map for warmup
                self.forward()
                loss = self.loss_fn(self.output.squeeze(1), self.label)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.loss = float(loss.detach().item())
                return
            
            self._optimize_parameters_e2e_pgd()
            return
        
        # Fallback to parent class
        super().optimize_parameters()
