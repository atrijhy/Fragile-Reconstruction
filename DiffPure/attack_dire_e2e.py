#!/usr/bin/env python3
# ---------------------------------------------------------------
# DIRE_Adv_Model with Linearization Attack
# 100% matching export_dire.py preprocessing and determinism
# ---------------------------------------------------------------

import argparse
import logging
import yaml
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from autoattack import AutoAttack
from stadv_eot.attacks import StAdvAttack

import utils
from typing import Optional
from utils import str2bool, get_accuracy, get_image_classifier, dict2namespace

# Import DIRE ODE model
from runners.dire_ode_model import DIRE_ODE_Model, ODEDireConfig


def conjugate_gradient_solve(jvp_fn, vjp_fn, rhs, x0=None, max_iters=20, lambda_reg=1e-4, tol=1e-6):
    """
    Matrix-free conjugate gradient solver for (J^T J + λI)δ = J^T Δf.
    
    Args:
        jvp_fn: function δ -> J·δ (Jacobian-vector product)
        vjp_fn: function v -> J^T·v (vector-Jacobian product)
        rhs: J^T Δf (the right-hand side vector)
        x0: initial guess (default: zero)
        max_iters: maximum CG iterations
        lambda_reg: Tikhonov regularization parameter λ
        tol: convergence tolerance
    
    Returns:
        δ: solution to the normal equation
    """
    device = rhs.device
    dtype = rhs.dtype
    
    # Initialize
    if x0 is None:
        x = torch.zeros_like(rhs)
    else:
        x = x0.clone()
    
    # Compute initial residual: r = b - Ax, where A = J^T J + λI
    # Ax = J^T(J·x) + λx
    Jx = jvp_fn(x)
    JTJx = vjp_fn(Jx)
    Ax = JTJx + lambda_reg * x
    r = rhs - Ax
    
    p = r.clone()
    rsold = torch.sum(r * r)
    
    if rsold.item() < tol:
        return x
    
    for i in range(max_iters):
        # Ap = (J^T J + λI)p
        Jp = jvp_fn(p)
        JTJp = vjp_fn(Jp)
        Ap = JTJp + lambda_reg * p
        
        # Step size
        pAp = torch.sum(p * Ap)
        alpha = rsold / (pAp + 1e-12)
        
        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * Ap
        
        rsnew = torch.sum(r * r)
        
        # Check convergence
        if rsnew.item() < tol:
            break
        
        # Update search direction
        beta = rsnew / (rsold + 1e-12)
        p = r + beta * p
        
        rsold = rsnew
    
    return x


def lsqr_solve(jvp_fn, vjp_fn, b, x0: Optional[torch.Tensor] = None, max_iters: int = 100, tol: float = 1e-6):
    """
    Matrix-free LSQR to solve min_x ||J x - b||_2 using only JVP (J·v) and VJP (J^T·u).
    This avoids forming J^T J explicitly and usually has better numerical properties
    than CG on normal equations. No Tikhonov damping here; prefer CG path for that.

    Args:
        jvp_fn: function v -> J·v, returns tensor shaped like b
        vjp_fn: function u -> J^T·u, returns tensor shaped like x
        b: right-hand side in output space (same shape as J·x), e.g. delta_f
        x0: optional initial guess (shape like input), zeros if None
        max_iters: maximum LSQR iterations
        tol: stopping tolerance based on |phi_bar|

    Returns:
        x: approximate solution (shape like input)
    """
    # Initialize
    device, dtype = b.device, b.dtype
    if x0 is None:
        # Build a zero vector in input space by applying vjp on zeros with same shape as b
        # Construct a dummy input by vjp_fn on a zero-like u
        # Safer: infer input shape via one VJP call
        u0 = torch.zeros_like(b)
        x_shape_template = vjp_fn(torch.zeros_like(u0))
        x = torch.zeros_like(x_shape_template)
    else:
        x = x0.clone()

    u = b.clone()
    beta = u.view(u.size(0), -1).norm(p=2, dim=1, keepdim=False) if u.dim() > 1 else u.norm()
    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if beta > 0:
        u = u / beta
    v = vjp_fn(u)
    alpha = v.view(v.size(0), -1).norm(p=2, dim=1, keepdim=False) if v.dim() > 1 else v.norm()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    if alpha > 0:
        v = v / alpha
    w = v.clone()
    phi_bar = float(beta)
    rho_bar = float(alpha)

    for _ in range(max_iters):
        # Bidiagonalization steps
        u = jvp_fn(v) - alpha * u
        beta = u.view(u.size(0), -1).norm(p=2, dim=1, keepdim=False) if u.dim() > 1 else u.norm()
        if isinstance(beta, torch.Tensor):
            beta = beta.item()
        if beta > 0:
            u = u / beta

        v = vjp_fn(u) - beta * v
        alpha = v.view(v.size(0), -1).norm(p=2, dim=1, keepdim=False) if v.dim() > 1 else v.norm()
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()
        if alpha > 0:
            v = v / alpha

        # Orthogonal transformation
        rho = (rho_bar ** 2 + beta ** 2) ** 0.5
        if rho == 0:
            break
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # Update solution and direction
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w

        # Convergence check (heuristic)
        if abs(phi_bar) < tol:
            break

    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def gmres_solve(jvp_fn, vjp_fn, b: torch.Tensor, x0: Optional[torch.Tensor] = None,
                restart: int = 50, max_iters: int = 200, tol: float = 1e-6) -> torch.Tensor:
    """
    Restarted matrix-free GMRES to solve J x = b using only JVP (J·v).
    vjp_fn is only used to infer input-shape when x0 is None.

    Args:
        jvp_fn: function v -> J·v (output-space tensor like b)
        vjp_fn: function u -> J^T·u (used once to infer input shape if x0 is None)
        b: RHS in output space
        x0: initial guess in input space (zeros if None)
        restart: GMRES restart parameter m (basis size)
        max_iters: total maximum iterations across restarts
        tol: residual tolerance on ||b - Jx||_2

    Returns:
        x: approximate solution in input space
    """
    with torch.no_grad():
        # Init x
        if x0 is None:
            # infer input-space shape by a single VJP on zeros
            u0 = torch.zeros_like(b)
            x_template = vjp_fn(u0)
            x = torch.zeros_like(x_template)
        else:
            x = x0.clone()

        def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.dot(a.view(-1), b.view(-1))

        # Initial residual
        r = b - jvp_fn(x)
        beta = r.norm().item()
        if beta < tol:
            return x

        iters = 0
        while iters < max_iters:
            m = min(restart, max_iters - iters)
            # Arnoldi storage
            V: list[torch.Tensor] = []
            H = torch.zeros((m + 1, m), device=b.device, dtype=b.dtype)
            # Givens cosine/sine
            cs = torch.zeros(m, device=b.device, dtype=b.dtype)
            sn = torch.zeros(m, device=b.device, dtype=b.dtype)
            # g vector
            g = torch.zeros(m + 1, device=b.device, dtype=b.dtype)
            g[0] = beta

            # v1
            v = r / beta
            V.append(v)

            j_converged = -1
            for j in range(m):
                w = jvp_fn(V[j])
                # Modified Gram-Schmidt
                for i in range(j + 1):
                    H[i, j] = dot(w, V[i])
                    w = w - H[i, j] * V[i]
                H[j + 1, j] = w.norm()
                if H[j + 1, j].item() != 0.0:
                    V.append(w / H[j + 1, j])
                else:
                    # happy breakdown
                    # pad remaining H entries as 0
                    pass

                # Apply previous Givens rotations
                for i in range(j):
                    temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                    H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                    H[i, j] = temp

                # Compute new Givens rotation
                denom = torch.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2) + 1e-12
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom
                # Apply to H and g
                H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
                H[j + 1, j] = b.new_tensor(0.0)
                g[j + 1] = -sn[j] * g[j]
                g[j] = cs[j] * g[j]

                resid = abs(g[j + 1].item())
                iters += 1
                if resid < tol or iters >= max_iters:
                    j_converged = j
                    break

            # Solve upper-triangular R y = g (size k where k=j_converged or m-1)
            k = j_converged if j_converged >= 0 else (m - 1)
            y = torch.zeros(k + 1, device=b.device, dtype=b.dtype)
            for i in range(k, -1, -1):
                s = g[i]
                for t in range(i + 1, k + 1):
                    s = s - H[i, t] * y[t]
                y[i] = s / (H[i, i] + 1e-12)

            # Update x
            for i in range(k + 1):
                x = x + y[i] * V[i]

            # Check outer residual if not converged
            if j_converged >= 0:
                return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            r = b - jvp_fn(x)
            beta = r.norm().item()
            if beta < tol:
                return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def build_jtj_and_rhs(jvp_fn, vjp_fn, rhs: torch.Tensor, input_shape: torch.Size,
                      device: Optional[torch.device] = None,
                      dtype: Optional[torch.dtype] = None,
                      max_n: int = 4096,
                      verbose: bool = True):
    """
    Build dense normal matrix A = J^T J and RHS vector b = J^T Δf for a single sample.
    
    DEPRECATED: This builds JTJ column-by-column via JVP/VJP (very slow).
    Use compute_jacobian_direct() instead for direct J computation.

    Args:
        jvp_fn: function v -> J·v, expects input-shaped tensor and returns DIRE-shaped tensor.
        vjp_fn: function u -> J^T·u, expects DIRE-shaped tensor and returns input-shaped tensor.
        rhs: J^T Δf as input-shaped tensor (same shape as a single input sample).
        input_shape: torch.Size of the input tensor (e.g., (1, C, H, W)). Batch must be 1.
        device: device to place the dense matrix on while building/solving.
        dtype: dtype to use for the dense matrix.
        max_n: safety cap on problem size (n = C*H*W). If exceeded, raise RuntimeError.
        verbose: print progress info.

    Returns:
        A: [n, n] tensor for (J^T J)
        b: [n] tensor for (J^T Δf)
        unflatten: callable to map a flat vector [n] back to input-shaped tensor
    """
    assert len(input_shape) == 4 and input_shape[0] == 1, "input_shape must be (1,C,H,W)"
    _, C, H, W = input_shape
    n = C * H * W
    if n > int(max_n):
        raise RuntimeError(f"Input dimensionality n={n} exceeds max_n={max_n} for dense JTJ build.")

    if device is None:
        device = rhs.device
    if dtype is None:
        dtype = rhs.dtype

    # Flatten helpers
    def flatten(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1)

    def unflatten(v: torch.Tensor) -> torch.Tensor:
        return v.view(1, C, H, W)

    # Prepare outputs
    A = torch.zeros((n, n), device=device, dtype=dtype)
    # Ensure rhs is on target device/dtype and flattened
    b = flatten(rhs.to(device=device, dtype=dtype)).clone()

    # Build columns of A by applying JTJ to basis vectors e_i
    # For memory efficiency and potential parallelism, we can batch a few basis vectors at a time.
    # Here we do it column-wise to keep implementation simple and robust.
    t_start = time.time()
    for i in range(n):
        # Basis vector e_i in input space
        ei = torch.zeros((1, C, H, W), device=device, dtype=dtype)
        # Compute spatial/channel indices
        c = (i // (H * W)) % C
        rem = i % (H * W)
        h = rem // W
        w = rem % W
        ei[0, c, h, w] = 1.0

        # Apply JTJ e_i = J^T(J e_i)
        Je = jvp_fn(ei)
        col = vjp_fn(Je)
        col = torch.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
        A[:, i] = flatten(col)

        if verbose and (i + 1) % max(1, (n // 8)) == 0:
            elapsed = time.time() - t_start
            print(f"    [JTJ build] {i+1}/{n} cols ({(i+1)/n*100:.1f}%) elapsed {elapsed:.1f}s")

    return A, b, unflatten


def compute_jacobian_direct(model_runner, x_sample, args, sample_idx, 
                            device: Optional[torch.device] = None,
                            dtype: Optional[torch.dtype] = None,
                            verbose: bool = True):
    """
    Directly compute the full Jacobian matrix J at x_sample using torch.func.jacfwd.
    Much faster than building JTJ column-by-column.
    
    Args:
        model_runner: DIRE_ODE_Model.runner with forward() method
        x_sample: input tensor [1, C, H, W] in [0,1]
        args: arguments with seed, t, eot_reps, fix_rand
        sample_idx: sample index for RNG seeding
        device: target device for J
        dtype: target dtype for J
        verbose: print timing info
        
    Returns:
        J: [m, n] Jacobian matrix where m=n=C*H*W (flattened)
        unflatten_input: function to reshape [n] -> [1,C,H,W]
        unflatten_output: function to reshape [m] -> [1,C,H,W]
    """
    if device is None:
        device = x_sample.device
    if dtype is None:
        dtype = x_sample.dtype
        
    _, C, H, W = x_sample.shape
    n = C * H * W
    
    # Define the DIRE forward function for jacfwd
    def dire_fn(x_flat):
        """Takes flattened input [n], returns flattened output [m]"""
        # Reshape to image
        x_img = x_flat.view(1, C, H, W)
        # [0,1] -> [-1,1]
        x_dire = (x_img - 0.5) * 2.0
        # Reset RNG for determinism
        torch.manual_seed(int(args.seed) + sample_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + sample_idx)
        # Run DIRE ODE
        out = model_runner.forward(
            x_dire,
            t_steps=int(args.t),
            eot_reps=int(args.eot_reps),
            fix_rand=bool(args.fix_rand)
        )
        dire_map = out['dire_map']  # [1, C, H, W] in [0,1]
        return dire_map.view(-1)
    
    # Flatten input
    x_flat = x_sample.view(-1).detach().requires_grad_(True)
    
    if verbose:
        print(f"  Computing Jacobian J ({n}×{n}) using torch.autograd.functional.jacobian...")
    
    t_start = time.time()
    
    # Use torch.autograd.functional.jacobian (compatible with odeint_adjoint)
    # Try vectorized mode when requested; fallback to non-vectorized if unsupported
    from torch.autograd.functional import jacobian
    vectorize_flag = bool(getattr(args, 'backproj_matrix_vectorize', False))
    try:
        J = jacobian(dire_fn, x_flat, create_graph=False, strict=False, vectorize=vectorize_flag)
    except Exception as e:
        if verbose and vectorize_flag:
            print(f"  [jacobian] vectorize=True failed ({e}); falling back to vectorize=False")
        J = jacobian(dire_fn, x_flat, create_graph=False, strict=False, vectorize=False)
    # Optional dtype downcast for speed/memory if specified
    target_dtype = dtype
    if target_dtype is None:
        # infer from args
        dtype_opt = str(getattr(args, 'backproj_matrix_dtype', 'fp32') or 'fp32').lower()
        if dtype_opt in ('fp16', 'half'):
            target_dtype = torch.float16
        elif dtype_opt in ('bf16', 'bfloat16'):
            target_dtype = torch.bfloat16
        else:
            target_dtype = J.dtype
    J = J.to(device=device, dtype=target_dtype)
    J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    
    elapsed = time.time() - t_start
    if verbose:
        bytes_per_el = int(getattr(J, 'element_size', lambda: 4)() if hasattr(J, 'element_size') else 4)
        mem_mb = J.numel() * bytes_per_el / 1e6
        print(f"  Jacobian computed in {elapsed:.2f}s, shape {J.shape}, memory {mem_mb:.1f} MB (dtype={J.dtype}, device={J.device})")
    
    def unflatten_input(v):
        return v.view(1, C, H, W)
    
    def unflatten_output(v):
        return v.view(1, C, H, W)
    
    return J, unflatten_input, unflatten_output


def _gaussian_smooth(delta: torch.Tensor, sigma: float = 0.5, ksize: int = 3) -> torch.Tensor:
    """Apply channel-wise Gaussian blur (depthwise) to BCHW delta for perceptual smoothing.
    Keeps gradients; runs on same device/dtype. No-op if sigma<=0 or ksize<=1.
    """
    if sigma <= 0 or ksize <= 1:
        return delta
    device, dtype = delta.device, delta.dtype
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    r = k // 2
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(x**2) / (2 * (float(sigma) ** 2)))
    g1 = g1 / g1.sum()
    g2 = g1[:, None] @ g1[None, :]
    kernel = g2[None, None, :, :]
    C = delta.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)
    padding = r
    return torch.nn.functional.conv2d(delta, kernel, stride=1, padding=padding, groups=C)


def make_deterministic(seed: int):
    """
    Configure environment + torch for best-effort determinism.
    EXACTLY copied from export_dire.py
    """
    # Environment variables (set before heavy libraries initialize where possible)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    # CUBLAS deterministic workspace config (helps reproducibility for some CUDA versions)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')

    # Seeds for python / numpy / torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # PyTorch determinism
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # older PyTorch or missing operator support may raise; we continue but issue a warning
        print('[warn] torch.use_deterministic_algorithms not available or raised.')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BinaryLogitsWrapper(nn.Module):
    """
    Wrapper to convert single-logit binary classifier to two-class format for AutoAttack.
    
    Single logit output: [batch_size] or [batch_size, 1]
    - logit > 0 → class 1 (fake)
    - logit < 0 → class 0 (real)
    
    Two-class output: [batch_size, 2]
    - logits[:, 0] = -logit (score for class 0/real)
    - logits[:, 1] = logit  (score for class 1/fake)
    
    This ensures: argmax(logits, dim=1) gives correct class prediction
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Get single logit output: can be [B] or [B, 1]
        logit = self.model(x)
        
        # Ensure logit is 2D [B, 1]
        if logit.dim() == 1:
            logit = logit.unsqueeze(1)
        
        # Convert to two-class format [B, 2]
        # Class 0 (real): -logit
        # Class 1 (fake): logit
        two_class_logits = torch.cat([-logit, logit], dim=1)
        
        return two_class_logits


class DIRE_Adv_Model(nn.Module):
    """
    DIRE Model with Linearization Attack.
    
    Preprocessing pipeline EXACTLY matching export_dire.py AND training:
    
    1. Load PNG images and apply transforms:
       tfm = Compose([
           Resize(256, InterpolationMode.BILINEAR),
           CenterCrop(256),
           ToTensor()
       ])
       → x01: [B, 3, 256, 256], [0, 1] range
    
    2. This x01 is the ATTACK INPUT SPACE
       Perturbations δ are added here: x_adv = x01 + δ
    
    3. Convert for DIRE: x = (x01 - 0.5) * 2.0 → [-1, 1]
       This happens inside forward, not in preprocessing
    
    4. DIRE outputs: dire_map [B, 3, 256, 256], [0, 1] range
    
    5. For classifier (matching training):
       - CenterCrop 256 → 224
       - Apply same normalization as training (ImageNet norm or scale_to_m11)
    
    Linearization:
    DIRE(x₀ + δ) ≈ DIRE(x₀) + ⟨∇DIRE(x₀), δ⟩
    where x₀, δ are in [0, 1] space, 256×256
    """
    
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        
        # Determine classifier preprocessing mode from args
        # IMPORTANT: For 'dire_resnet*' loaders, the returned classifier wrapper already
        # performs input scaling to [-1,1] internally (default). Therefore we MUST NOT
        # apply ImageNet normalization here, otherwise inputs get double-normalized.
        self.apply_imagenet_norm = getattr(args, 'dire_apply_imagenet_norm', True)
        self.scale_to_m11 = getattr(args, 'dire_scale_to_m11', False) and not self.apply_imagenet_norm
        
        # ImageNet normalization parameters (stored as lists, create tensors on-device)
        self._imagenet_mean = [0.485, 0.456, 0.406]
        self._imagenet_std = [0.229, 0.224, 0.225]
        
        # IMPORTANT: We handle normalization in preprocess_for_classifier to match training.
        # So we load the UNWRAPPED classifier (use '_unwrapped' suffix to prevent double normalization)
        classifier_name = args.classifier_name
        if 'dire_resnet' in classifier_name and '_unwrapped' not in classifier_name:
            # Add '_unwrapped' to prevent the classifier loader from applying normalization
            # since we do it ourselves in preprocess_for_classifier
            print("[DIRE_Adv_Model] Note: We'll apply normalization in preprocess_for_classifier")
            print("                Loading base model without wrapper normalization")
        
        # If using dire_resnet*, decide wrapper variant:
        # - If args.dire_apply_imagenet_norm=True, request '_imnet' variant so normalization happens INSIDE wrapper
        # - Otherwise use default M11 variant (wrapper scales to [-1,1])
        classifier_name = args.classifier_name
        if classifier_name.startswith('dire_resnet'):
            if bool(getattr(args, 'dire_apply_imagenet_norm', True)):
                # Ensure we get the ImageNet-norm wrapper
                if '_imnet' not in classifier_name:
                    classifier_name = classifier_name + '_imnet'
                print(f"[DIRE_Adv_Model] Using dire_resnet wrapper with ImageNet normalization (name='{classifier_name}')")
            elif bool(getattr(args, 'dire_scale_to_m11', True)):
                if '_m11' not in classifier_name:
                    classifier_name = classifier_name + '_m11'
                print(f"[DIRE_Adv_Model] Using dire_resnet wrapper with [−1,1] scaling (name='{classifier_name}')")
            else:
                print(f"[DIRE_Adv_Model] Using dire_resnet wrapper with [0,1] scaling (default)")
            # Disable external preprocessing to avoid double-normalization; wrapper handles it.
            self.apply_imagenet_norm = False
            self.scale_to_m11 = False

        print(f"[DIRE_Adv_Model] Classifier preprocessing (effective):")
        print(f"  apply_imagenet_norm: {self.apply_imagenet_norm}")
        print(f"  scale_to_m11: {self.scale_to_m11}")

        self.classifier = get_image_classifier(
            classifier_name,
            ckpt_path=getattr(args, 'classifier_ckpt', None)
        ).to(config.device)
        
        # DIRE ODE model configuration (EXACTLY matching export_dire.py)
        print(f'[DIRE] Initializing ODE model with t={args.t}')
        ode_cfg = ODEDireConfig(
            t_steps=int(args.t),
            method=str(args.ode_method),
            rtol=float(args.ode_rtol),
            atol=float(args.ode_atol),
            step_size=float(args.ode_step_size),
            fix_rand=bool(args.fix_rand),
            eot_reps=int(args.eot_reps),
            clamp_recon=True,
            dire_scale_to_01=True,
            dire_smooth_eps=float(args.ode_dire_smooth_eps),
            dire_log_grad=False,
            force_fp32=True,
            nan_fallback=True,
            use_unet_checkpoint=True,
            disable_solver_fallback=True,
            local_half=False,  # <<== disabled for deterministic fp32 export
            freeze_model_params=False,
            empty_cache_each_iter=True,
        )
        
        self.runner = DIRE_ODE_Model(args, config, ode_cfg=ode_cfg, device=config.device)
        self.runner.eval()
        
        # Clear cached noise (matching export_dire.py)
        try:
            if hasattr(self.runner, '_fixed_noise'):
                self.runner._fixed_noise = None
            if hasattr(self.runner, '_fixed_noise_cache'):
                self.runner._fixed_noise_cache = {}
        except Exception:
            pass
        
        # Print config summary (matching export_dire.py)
        try:
            print('[export][ode] cfg:', {
                't_steps': ode_cfg.t_steps,
                'method': ode_cfg.method,
                'step_size': ode_cfg.step_size,
                'rtol': ode_cfg.rtol,
                'atol': ode_cfg.atol,
                'fix_rand': ode_cfg.fix_rand,
                'eot_reps': ode_cfg.eot_reps,
                'dire_smooth_eps': ode_cfg.dire_smooth_eps,
            })
        except Exception:
            pass
        
        # Linearization buffers
        self.x_original = None  # Original x01: 256×256, [0, 1]
        self.dire_base = None  # DIRE(x₀): 256×256, [0, 1]
        self.dire_gradient = None  # ∇DIRE(x₀): same shape as x_original
        
        self.mode = 'full'
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None
    
    def reset_counter(self):
        self.counter.zero_()
    
    def set_tag(self, tag=None):
        self.tag = tag
    
    def set_mode(self, mode):
        assert mode in ['full', 'linear', 'dire_space'], f"Mode must be 'full', 'linear', or 'dire_space', got {mode}"
        self.mode = mode
        print(f"[DIRE_Adv_Model] mode set to: {mode}")
    
    def preprocess_for_classifier(self, dire_output):
        """
                Preprocess DIRE output for classifier.
                Match training preprocessing:
                - For 256-based pipelines (ImageNet style): upsample to 256 then CenterCrop(224), then norm.
                - For native small-res pipelines (e.g., CIFAR10 32x32 with dire_base_256=False, target_size=32):
                    keep native resolution (no upsample, no 224 crop); optionally resize to target if needed; norm handled by wrapper.
        
        Args:
            dire_output: [B, 3, H, W], [0, 1] range (H=W, e.g., 64 or 256)
        
        Returns:
            classifier_input: [B, 3, 224, 224], normalized appropriately
        """
        tensor = dire_output
        img_size = int(getattr(self.args, 'dire_image_size', 256))

        # Case A: native small-res (e.g., 32x32) → keep native size, no 224 crop
        if img_size <= 64:
            if tensor.shape[-2:] != (img_size, img_size):
                tensor = TF.resize(tensor, [img_size, img_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
            # Normalization here only if enabled (for non-dire_resnet wrappers). For dire_resnet, wrapper handles it.
            if self.apply_imagenet_norm:
                mean = torch.tensor(self._imagenet_mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
                std = torch.tensor(self._imagenet_std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
                tensor = (tensor - mean) / std
            elif self.scale_to_m11:
                tensor = tensor * 2.0 - 1.0
            return tensor

        # Case B: 256-based pipeline (default ImageNet-style)
        if tensor.shape[-2:] != (256, 256):
            tensor = TF.resize(tensor, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
        tensor = TF.center_crop(tensor, 224)

        if self.apply_imagenet_norm:
            mean = torch.tensor(self._imagenet_mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
            std = torch.tensor(self._imagenet_std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
            tensor = (tensor - mean) / std
        elif self.scale_to_m11:
            tensor = tensor * 2.0 - 1.0
        return tensor
    
    def compute_dire_and_gradient(self, x, batch_size=1):
        """
        Compute DIRE(x₀) for CG-based backprojection.
        No longer computes diagonal Jacobian - we use JVP/VJP operators directly.
        
        Args:
            x: [B, 3, 256, 256], [0, 1] range (output of tfm in export_dire.py)
            batch_size: Process images in mini-batches (default=1 to avoid OOM)
        
        Returns:
            dire_output: [B, 3, 256, 256], [0, 1] range
            dire_grad: None (not used in CG mode)
        """
        print("=" * 70)
        print("Computing DIRE transformation (CG mode - no diagonal estimation)")
        print("=" * 70)
        start_time = time.time()
        
        total_samples = x.shape[0]
        print(f"[compute_dire_and_gradient]")
        print(f"  Total samples: {total_samples}")
        print(f"  Mini-batch size: {batch_size}")
        print(f"  Input shape: {x.shape}")
        print(f"  Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        
        # Verify input is expected size, [0, 1]
        expected_size = int(getattr(self.args, 'dire_image_size', 256))
        assert x.shape[-2:] == (expected_size, expected_size), \
            f"Expected {expected_size}×{expected_size} input, got {x.shape[-2:]}"
        assert x.min() >= 0 and x.max() <= 1, f"Expected [0, 1] range, got [{x.min():.4f}, {x.max():.4f}]"
        
        # Store original input (attack space) - ensure it's on the correct device
        self.x_original = x.detach().clone().to(self.config.device)
        
        # Process in mini-batches to avoid OOM
        all_dire_outputs = []
        
        num_batches = (total_samples + batch_size - 1) // batch_size
        print(f"  Processing in {num_batches} mini-batches...")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            
            print(f"\n  Mini-batch {i+1}/{num_batches} (samples {start_idx}-{end_idx-1})")
            
            # Get mini-batch and ensure it's on the correct device
            x_batch = x[start_idx:end_idx].to(self.config.device).detach()
            
            # Convert to DIRE input space: [0, 1] → [-1, 1]
            x_dire_input = (x_batch - 0.5) * 2.0
            
            # Reset RNG for determinism
            torch.manual_seed(int(self.args.seed) + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.args.seed) + i)
            
            # Compute DIRE transformation
            batch_start = time.time()
            with torch.no_grad():
                out = self.runner.forward(
                    x_dire_input,
                    t_steps=int(self.args.t),
                    eot_reps=int(self.args.eot_reps),
                    fix_rand=bool(self.args.fix_rand)
                )
                dire_output_batch = out['dire_map']  # [mini_B, C, H, W] in [0,1]
            # Report non-finite if any (helps diagnose ODE fallback)
            if not torch.isfinite(dire_output_batch).all():
                print("  [warn] Non-finite values detected in DIRE output (mini-batch)")
            
            batch_time = time.time() - batch_start
            print(f"    DIRE computation: {batch_time:.2f}s")
            
            # Store results
            all_dire_outputs.append(dire_output_batch.detach())
            
            # Free memory
            del x_batch, x_dire_input, dire_output_batch
            torch.cuda.empty_cache()
        
        # Concatenate all batches
        dire_output = torch.cat(all_dire_outputs, dim=0)
        
        print(f"\n  All mini-batches completed!")
        print(f"  Final DIRE output shape: {dire_output.shape}")
        
        # Store for linearization
        self.dire_base = dire_output
        self.dire_gradient = None  # Not used in CG mode
        
        total_time = time.time() - start_time
        print("=" * 70)
        print(f"Total computation time: {total_time:.2f}s ({total_time/total_samples:.2f}s per sample)")
        print("DIRE output ready for CG backprojection!")
        print("=" * 70 + "\n")
        
        return dire_output, None
    
    def forward_linear(self, x):
        """
        Forward pass using linearized DIRE approximation.
        
        DEPRECATED: Diagonal Jacobian approximation has been removed.
        Use METHOD 2 (full) or METHOD 3 (dire_space with CG backprojection) instead.
        
        Args:
            x: [B, 3, 256, 256], [0, 1] range (potentially adversarial)
        
        Returns:
            out: classifier output
        """
        raise NotImplementedError(
            "METHOD 1 (linear with diagonal Jacobian) has been deprecated.\n"
            "Diagonal approximation is too inaccurate and has been removed.\n"
            "Please use:\n"
            "  - METHOD 2 (--attack_method full): Direct attack on complete pipeline\n"
            "  - METHOD 3 (--attack_method dire_space): Attack in DIRE space + CG backprojection"
        )
    
    def forward_dire_space(self, dire_map):
        """
        Forward pass directly on DIRE output space (for DIRE-space attack).
        
        This mode allows attacking the classifier in DIRE output space,
        then backprojecting perturbations to input space using gradient.
        
        Args:
            dire_map: [B, 3, 256, 256], [0, 1] range (DIRE output, possibly perturbed)
        
        Returns:
            out: classifier output
        """
        # Preprocess for classifier (matching training)
        # This includes: crop 256→224 + normalization
        classifier_input = self.preprocess_for_classifier(dire_map)
        if self.tag in ('attack_dire_space', 'grad_probe'):
            try:
                print(f"  [debug] dire_map.requires_grad={getattr(dire_map, 'requires_grad', None)}")
                print(f"  [debug] classifier_input.requires_grad={getattr(classifier_input, 'requires_grad', None)}")
            except Exception:
                pass
        
        # Pass to classifier
        out = self.classifier(classifier_input)
        if self.tag in ('attack_dire_space', 'grad_probe'):
            try:
                print(f"  [debug] classifier_logit.requires_grad={getattr(out, 'requires_grad', None)}")
            except Exception:
                pass
        
        return out
    
    def forward_full(self, x):
        """
        Full forward pass with actual DIRE transformation.
        
        Args:
            x: [B, 3, 256, 256], [0, 1] range
        
        Returns:
            out: classifier output
        """
        counter = self.counter.item()
        
        _quiet_full = bool(getattr(self.args, 'quiet_full', False))
        #if (not _quiet_full) and (counter % 5 == 0):
            #print(f'[Full DIRE] Forward pass #{counter}')
            #print(f'  Input: {x.shape}, range: [{x.min().item():.4f}, {x.max().item():.4f}]')
        
        # Verify input
        expected_size = int(getattr(self.args, 'dire_image_size', 256))
        assert x.shape[-2:] == (expected_size, expected_size), \
            f"Expected {expected_size}×{expected_size} input, got {x.shape[-2:]}"
        
    # Convert to DIRE input space: [0, 1] → [-1, 1]
        # EXACTLY matching export_dire.py: x = (x01 - 0.5) * 2.0
        x_dire_input = (x - 0.5) * 2.0
        
        #if (not _quiet_full) and (counter % 5 == 0):
            #print(f'  After preprocessing: {x_dire_input.shape}, '
                  #f'range: [{x_dire_input.min().item():.4f}, {x_dire_input.max().item():.4f}]')
        
        # Reset RNG for determinism (matching export_dire.py)
        torch.manual_seed(int(self.args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.args.seed))
        
        start_time = time.time()
        
        # Apply full DIRE transformation
        out = self.runner.forward(
            x_dire_input,
            t_steps=int(self.args.t),
            eot_reps=int(self.args.eot_reps),
            fix_rand=bool(self.args.fix_rand)
        )
        dire_output = out['dire_map']  # [B, C, H, W] in [0,1]
        
        elapsed = time.time() - start_time

        #if (not _quiet_full) and (counter % 5 == 0):
            #print(f'  DIRE output: {dire_output.shape}, '
                  #f'range: [{dire_output.min().item():.4f}, {dire_output.max().item():.4f}]')
            #print(f'  DIRE time: {elapsed:.2f}s')

        # Preprocess for classifier (matching training)
        # This includes: crop 256→224 + normalization
        classifier_input = self.preprocess_for_classifier(dire_output)
        
        #if (not _quiet_full) and (counter % 5 == 0):
            #print(f'  Classifier input: {classifier_input.shape}, '
                  #f'range: [{classifier_input.min().item():.4f}, {classifier_input.max().item():.4f}]')

        # Pass to classifier
        out = self.classifier(classifier_input)
        
        self.counter += 1
        
        return out
    
    def forward(self, x):
        """Main forward pass - dispatches to full, linear, or dire_space mode."""
        if self.mode == 'full':
            return self.forward_full(x)
        elif self.mode == 'linear':
            return self.forward_linear(x)
        elif self.mode == 'dire_space':
            return self.forward_dire_space(x)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def load_data_with_export_transforms(args):
    """
    Load PNG images and apply EXACTLY the same transforms as export_dire.py
    
    Returns images in [B, 3, 256, 256], [0, 1] range
    This is the ATTACK INPUT SPACE
    """
    from pathlib import Path
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    
    # EXACTLY matching export_dire.py transform, but parameterized by dire_image_size
    img_size = int(getattr(args, 'dire_image_size', 256))
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    class SimpleImageDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            x = self.transform(img)  # [C,H,W], [0,1]
            return x, self.labels[idx]
    
    # Use the custom data loading from args if available
    if hasattr(args, 'data_root') and args.data_root:
        # Custom data loading path
        from utils import load_data
        x_val, y_val = load_data(args, args.adv_batch_size)
        return x_val, y_val
    
    # Fallback: use standard load_data from utils
    from utils import load_data
    x_val, y_val = load_data(args, args.adv_batch_size)
    
    return x_val, y_val


def eval_autoattack_dire(
    args, config, model, x_val, y_val, adv_batch_size, log_dir,
    return_summary: bool = False,
    suppress_save: bool = False,
    suppress_final_print: bool = False,
    stream_batch_idx: Optional[int] = None,
    global_start_idx: int = 0,
    progress_prefix: Optional[dict] = None,
):
    """
    Evaluate AutoAttack on DIRE model with linearization strategy.
    
    Args:
        global_start_idx: Starting index for this batch in the full dataset (for sample naming & progress).
    """
    # No need to check for DataParallel since we disabled it
    model_ = model
    
    attack_version = args.attack_version
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'rand':
        attack_list = ['apgd-ce', 'apgd-dlr']
    elif attack_version == 'custom':
        attack_list = args.attack_type.split(',')
    else:
        raise NotImplementedError(f'Unknown attack version: {attack_version}!')
    
    print("\n" + "=" * 70)
    print("AUTOATTACK EVALUATION ON DIRE CLASSIFIER")
    print(f"Attack version: {attack_version}")
    print(f"Attack list: {attack_list}")
    print(f"Attack method: {args.attack_method}")
    print(f"Norm: {args.lp_norm}, Epsilon: {args.adv_eps}")
    print(f"Processing {x_val.shape[0]} samples one-by-one to avoid OOM")
    print("=" * 70 + "\n")
    
    # Prepare to collect results for both attack methods
    total_samples = x_val.shape[0]
    
    # Determine which methods to run
    run_linear = args.attack_method in ['linear', 'both']
    run_full = args.attack_method in ['full', 'both']
    run_dire_space = args.attack_method == 'dire_space'
    run_zeroth = args.attack_method == 'zeroth'
    
    # Method 1: Linearized attack
    all_x_adv_linear = []
    linear_clean_correct = 0
    linear_adv_correct = 0
    
    # Method 2: Full DIRE attack
    all_x_adv_full = []
    full_attack_times = []  # per-sample attack time (seconds)
    full_clean_correct = 0
    full_adv_correct = 0
    # Track how many of the clean-correct samples remain correct after attack (for conditional success)
    full_adv_correct_on_clean = 0
    # Per-class counters for METHOD 2
    full_total_real = 0
    full_total_fake = 0
    full_clean_correct_real = 0
    full_clean_correct_fake = 0
    full_adv_correct_real = 0
    full_adv_correct_fake = 0
    full_adv_correct_on_clean_real = 0
    full_adv_correct_on_clean_fake = 0
    
    # Method 3: DIRE-space attack
    all_x_adv_dire_space = []
    dire_space_total_attack_times = []  # per-sample total time (autoattack + CG)
    dire_space_clean_correct = 0  # clean acc in DIRE space
    # Two-stage robust acc counters for DIRE-space attack (Stage2/linear removed)
    dire_space_stage1_adv_correct = 0  # after attacking classifier in DIRE space (classifier-only)
    dire_space_stage3_adv_correct = 0  # after backprojecting to input, evaluated with full pipeline
    # Additionally track conditional on full-pipeline clean-correct subset
    dire_space_full_clean_correct = 0
    dire_space_stage3_adv_correct_on_cleanfull = 0
    # Per-class counters for METHOD 3 (Stage3/full-pipeline)
    dire_stage3_total_real = 0
    dire_stage3_total_fake = 0
    dire_space_full_clean_correct_real = 0
    dire_space_full_clean_correct_fake = 0
    dire_space_stage3_adv_correct_real = 0
    dire_space_stage3_adv_correct_fake = 0
    dire_space_stage3_adv_correct_on_cleanfull_real = 0
    dire_space_stage3_adv_correct_on_cleanfull_fake = 0

    # Method 4: Zeroth-order black-box attack (input-space)
    all_x_adv_zeroth = []
    zeroth_attack_times = []  # per-sample attack time (seconds)
    # Optional running prefixes for streaming progress (cumulative counts from previous batches)
    pp = progress_prefix or {}
    pp_linear_clean = int(pp.get('linear_clean_correct', 0))
    pp_linear_adv = int(pp.get('linear_adv_correct', 0))
    pp_full_clean = int(pp.get('full_clean_correct', 0))
    pp_full_adv = int(pp.get('full_adv_correct', 0))
    pp_full_on_clean = int(pp.get('full_adv_correct_on_clean', 0))
    pp_full_total_real = int(pp.get('full_total_real', 0))
    pp_full_total_fake = int(pp.get('full_total_fake', 0))
    pp_full_clean_real = int(pp.get('full_clean_correct_real', 0))
    pp_full_clean_fake = int(pp.get('full_clean_correct_fake', 0))
    pp_full_adv_real = int(pp.get('full_adv_correct_real', 0))
    pp_full_adv_fake = int(pp.get('full_adv_correct_fake', 0))
    pp_full_on_clean_real = int(pp.get('full_adv_correct_on_clean_real', 0))
    pp_full_on_clean_fake = int(pp.get('full_adv_correct_on_clean_fake', 0))
    # DIRE-space prefixes
    pp_ds_clean = int(pp.get('dire_space_clean_correct', 0))
    pp_ds_stage1 = int(pp.get('dire_space_stage1_adv_correct', 0))
    pp_ds_stage3 = int(pp.get('dire_space_stage3_adv_correct', 0))
    pp_ds_full_clean = int(pp.get('dire_space_full_clean_correct', 0))
    pp_ds_on_cleanfull = int(pp.get('dire_space_stage3_adv_correct_on_cleanfull', 0))
    pp_ds_total_real = int(pp.get('dire_stage3_total_real', 0))
    pp_ds_total_fake = int(pp.get('dire_stage3_total_fake', 0))
    pp_ds_full_clean_real = int(pp.get('dire_space_full_clean_correct_real', 0))
    pp_ds_full_clean_fake = int(pp.get('dire_space_full_clean_correct_fake', 0))
    pp_ds_stage3_real = int(pp.get('dire_space_stage3_adv_correct_real', 0))
    pp_ds_stage3_fake = int(pp.get('dire_space_stage3_adv_correct_fake', 0))
    pp_ds_on_cleanfull_real = int(pp.get('dire_space_stage3_adv_correct_on_cleanfull_real', 0))
    pp_ds_on_cleanfull_fake = int(pp.get('dire_space_stage3_adv_correct_on_cleanfull_fake', 0))
    # Zeroth prefixes
    pp_z_clean = int(pp.get('zeroth_clean_correct', 0))
    pp_z_adv = int(pp.get('zeroth_adv_correct', 0))
    pp_z_on_clean = int(pp.get('zeroth_adv_correct_on_clean', 0))
    pp_z_total_real = int(pp.get('zeroth_total_real', 0))
    pp_z_total_fake = int(pp.get('zeroth_total_fake', 0))
    pp_z_clean_real = int(pp.get('zeroth_clean_correct_real', 0))
    pp_z_clean_fake = int(pp.get('zeroth_clean_correct_fake', 0))
    pp_z_adv_real = int(pp.get('zeroth_adv_correct_real', 0))
    pp_z_adv_fake = int(pp.get('zeroth_adv_correct_fake', 0))
    pp_z_on_clean_real = int(pp.get('zeroth_adv_correct_on_clean_real', 0))
    pp_z_on_clean_fake = int(pp.get('zeroth_adv_correct_on_clean_fake', 0))

    zeroth_clean_correct = 0
    zeroth_adv_correct = 0
    zeroth_adv_correct_on_clean = 0
    # Per-class counters for METHOD 4
    zeroth_total_real = 0
    zeroth_total_fake = 0
    zeroth_clean_correct_real = 0
    zeroth_clean_correct_fake = 0
    zeroth_adv_correct_real = 0
    zeroth_adv_correct_fake = 0
    zeroth_adv_correct_on_clean_real = 0
    zeroth_adv_correct_on_clean_fake = 0
    
    # Wrap DIRE model for AutoAttack compatibility
    model_wrapped = BinaryLogitsWrapper(model)
    
    # Prepare adv image save dirs
    if getattr(args, 'save_adv_images', False):
        # Distinguish by classifier and fake source so that
        # different (train,test) pairs do not overwrite each other.
        clf_tag = str(getattr(args, 'classifier_name', 'clf'))
        fake_tag_raw = str(getattr(args, 'fake_dirs', '') or '')
        if fake_tag_raw:
            fake_tag = fake_tag_raw.replace('/', '_').replace(',', '+')
        else:
            fake_tag = 'unknown'
        tag_suffix = f"{clf_tag}_{fake_tag}"

        save_dir_full = os.path.join(log_dir, 'adv_images_rerun', f'full_{tag_suffix}')
        save_dir_dire_space = os.path.join(log_dir, 'adv_images_rerun', f'dire_space_{tag_suffix}')
        save_dir_zeroth = os.path.join(log_dir, 'adv_images_rerun', f'zeroth_{tag_suffix}')

        os.makedirs(save_dir_full, exist_ok=True)
        os.makedirs(save_dir_dire_space, exist_ok=True)
        os.makedirs(save_dir_zeroth, exist_ok=True)

    # Lightweight gradient probe to detect broken grad chains before launching attacks
    def _grad_probe(x_in: torch.Tensor, y_in: torch.Tensor, mode_name: str) -> None:
        try:
            model_.set_mode(mode_name)
            model_.set_tag('grad_probe')
            x_probe = x_in.detach().clone().to(config.device).requires_grad_(True)
            with torch.enable_grad():
                if mode_name == 'dire_space':
                    # Probe internals to see where gradient vanishes
                    x_cls = model_.preprocess_for_classifier(x_probe)
                    x_cls.retain_grad()
                    one_logit = model_.classifier(x_cls)  # shape [B] or [B,1]
                    if one_logit.dim() == 1:
                        one_logit = one_logit.unsqueeze(1)
                    logits2 = torch.cat([-one_logit, one_logit], dim=1)
                else:
                    logits2 = model_wrapped(x_probe)
                    x_cls = None
                # expect [B,2]
                if logits2.dim() != 2 or logits2.size(1) != 2:
                    print(f"  [grad-probe] unexpected logits shape: {tuple(logits2.shape)} (expect [B,2])")
                loss = F.cross_entropy(logits2, y_in.to(config.device))
                loss.backward()
                if x_probe.grad is None:
                    print("  [grad-probe] x.grad is None → gradient path is broken")
                else:
                    g = torch.nan_to_num(x_probe.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    print(f"  [grad-probe] dL/d(input) Linf={g.abs().max().item():.3e}, L2={g.norm().item():.3e}")
                    save_grad_dir = str(getattr(args, 'save_grad_probe_dir', '') or '')
                    if save_grad_dir:
                        os.makedirs(save_grad_dir, exist_ok=True)
                        grad_path = os.path.join(
                            save_grad_dir,
                            f"sample_{int(global_sample_idx):06d}_{mode_name}_grad.pt",
                        )
                        torch.save({
                            "grad": g.detach().cpu(),
                            "input": x_in.detach().cpu(),
                            "label": y_in.detach().cpu(),
                            "global_sample_idx": int(global_sample_idx),
                            "mode": mode_name,
                            "classifier_ckpt": getattr(args, 'classifier_ckpt', None),
                            "fake_dirs": getattr(args, 'fake_dirs', None),
                        }, grad_path)
                if x_cls is not None and x_cls.grad is not None:
                    gc = torch.nan_to_num(x_cls.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    print(f"  [grad-probe] dL/d(crop+norm) Linf={gc.abs().max().item():.3e}, L2={gc.norm().item():.3e}")
        except Exception as e:
            print(f"  [grad-probe] exception: {e}")
        finally:
            try:
                model_.set_tag(None)
            except Exception:
                pass

    # Decide processing order: attack fake samples first, then real samples.
    # This only affects the order in which we launch attacks / save outputs;
    # statistics (counts, accuracies) are unchanged.
    labels_cpu = y_val.detach().cpu().view(-1).tolist()
    fake_label = int(getattr(args, 'fake_label', 1))
    real_label = int(getattr(args, 'real_label', 0))
    fake_indices = [i for i, y in enumerate(labels_cpu) if y == fake_label]
    real_indices = [i for i, y in enumerate(labels_cpu) if y == real_label]
    other_indices = [i for i, y in enumerate(labels_cpu) if y not in (fake_label, real_label)]
    ordered_indices = fake_indices + real_indices + other_indices

    # Process each sample individually (in the chosen order)
    for local_idx, sample_idx in enumerate(ordered_indices):
        # Compute global sample index for this batch (monotone in processing order)
        global_sample_idx = global_start_idx + local_idx
        
        print(f"\n{'=' * 70}")
        print(f"Processing sample {global_sample_idx + 1} (batch local: {sample_idx + 1}/{total_samples})")
        print(f"{'=' * 70}")
        
        # Get single sample
        x_sample = x_val[sample_idx:sample_idx+1].to(config.device)
        y_sample = y_val[sample_idx:sample_idx+1].to(config.device)
        
        # ====================================================================
        # METHOD 1: Linearized DIRE Attack
        # ====================================================================
        if run_linear:
            print("\n" + "=" * 70)
            print("METHOD 1: Linearized DIRE Attack (Fast)")
            print("=" * 70)
            
            # Step 1: Compute DIRE and gradient for this sample
            print("\n[Step 1] Computing DIRE and gradient...")
            model_.reset_counter()
            model_.set_tag('compute_gradient')
            model_.set_mode('full')
            
            # Compute DIRE(x) and ∇DIRE(x) for this single sample
            dire_output, dire_gradient = model_.compute_dire_and_gradient(x_sample, batch_size=1)
            
            # Verify linearization quality
            model_.set_mode('linear')
            with torch.no_grad():
                linear_pred = model(x_sample)
                linear_logit = linear_pred.view(-1)
                linear_class = (linear_logit > 0).long()
                
                model_.set_mode('full')
                full_pred = model(x_sample)
                full_logit = full_pred.view(-1)
                full_class = (full_logit > 0).long()
                
                linear_correct = (linear_class == y_sample).item()
                full_correct = (full_class == y_sample).item()
                linear_clean_correct += full_correct  # Track clean accuracy for METHOD 1
                
                print(f"  Linear prediction: {linear_class.item()} (correct: {linear_correct})")
                print(f"  Full prediction: {full_class.item()} (correct: {full_correct})")
            
            # ====================================================================
            # Step 2: Attack this sample
            # ====================================================================
            print("\n[Step 2] Attacking linearized DIRE...")
            model_.reset_counter()
            model_.set_mode('linear')
            model_.set_tag('attack')
            
            # Create AutoAttack for this sample
            adversary = AutoAttack(
                model_wrapped,
                norm=args.lp_norm,
                eps=args.adv_eps,
                version=attack_version,
                attacks_to_run=attack_list,
                log_path=f'{log_dir}/log_sample_{sample_idx}.txt',
                device=config.device,
                verbose=False  # Reduce output clutter
            )
            
            if attack_version == 'custom':
                adversary.apgd.n_restarts = 1
                adversary.apgd.n_iter = int(getattr(args, 'autoattack_n_iter', 100))
            
            # Attack this single sample
            x_adv_sample = adversary.run_standard_evaluation(x_sample, y_sample, bs=1)
            
            # Check if perturbation was actually applied
            perturbation = (x_adv_sample - x_sample).abs()
            pert_linf = perturbation.max().item()
            pert_l2 = perturbation.norm().item()
            print(f"  Perturbation: L∞={pert_linf:.6f}, L2={pert_l2:.6f}")
            
            if pert_linf < 1e-6:
                print(f"  [WARNING] No perturbation detected! Attack may have failed.")
            
            # ====================================================================
            # Step 3: Evaluate robustness on full DIRE
            # ====================================================================
            if not getattr(args, 'quiet_full', False):
                print("\n[Step 3] Evaluating on full DIRE...")
            model_.reset_counter()
            model_.set_mode('full')
            model_.set_tag('eval_adv')
            
            with torch.no_grad():
                # Get prediction on adversarial example
                adv_pred = model(x_adv_sample)
                adv_logit = adv_pred.view(-1)
                adv_class = (adv_logit > 0).long()
                adv_correct = (adv_class == y_sample).item()
                linear_adv_correct += adv_correct
                
                print(f"  Adversarial prediction: {adv_class.item()} (correct: {adv_correct})")
                print(f"  Attack {'failed' if adv_correct else 'succeeded'}!")
            
            # Store linearized adversarial example
            all_x_adv_linear.append(x_adv_sample.cpu())
        
        # ====================================================================
        # METHOD 2: Full DIRE Attack (Slow but Accurate)
        # ====================================================================
        if run_full:
            if not getattr(args, 'quiet_full', False):
                print("\n" + "=" * 70)
                print("METHOD 2: Full DIRE Attack (No Linearization)")
                print("=" * 70)
            
            # Step 1: Verify clean prediction
            if not getattr(args, 'quiet_full', False):
                print("\n[Step 1] Checking clean prediction...")
            model_.reset_counter()
            model_.set_mode('full')
            model_.set_tag('clean_full')
            
            with torch.no_grad():
                clean_pred = model(x_sample)
                clean_logit = clean_pred.view(-1)
                clean_class = (clean_logit > 0).long()
                clean_correct = (clean_class == y_sample).item()
                full_clean_correct += clean_correct
                # per-class updates
                lbl = int(y_sample.item())
                if lbl == int(args.real_label):
                    full_total_real += 1
                    full_clean_correct_real += clean_correct
                elif lbl == int(args.fake_label):
                    full_total_fake += 1
                    full_clean_correct_fake += clean_correct
                
                if not getattr(args, 'quiet_full', False):
                    print(f"  Clean prediction: {clean_class.item()} (correct: {clean_correct})")
            
            # Gradient probe before launching the full attack
            if not getattr(args, 'quiet_full', False):
                _grad_probe(x_sample, y_sample, mode_name='full')
            elif getattr(args, 'save_grad_probe_dir', None):
                _grad_probe(x_sample, y_sample, mode_name='full')

            if getattr(args, 'grad_probe_only', False):
                continue

            # Step 2: Direct attack on full DIRE (no gradient precomputation)
            if not getattr(args, 'quiet_full', False):
                print("\n[Step 2] Attacking full DIRE (this will be slow)...")
            model_.reset_counter()
            model_.set_mode('full')  # Always use complete ODE solver
            model_.set_tag('attack_full')
            
            # Create new AutoAttack instance for full model
            adversary_full = AutoAttack(
                model_wrapped,
                norm=args.lp_norm,
                eps=args.adv_eps,
                version=attack_version,
                attacks_to_run=attack_list,
                log_path=f'{log_dir}/log_sample_{sample_idx}_full.txt',
                device=config.device,
                verbose=False
            )
            
            if attack_version == 'custom':
                adversary_full.apgd.n_restarts = 1  # Single run for fair timing comparison
                adversary_full.apgd.n_iter = int(getattr(args, 'autoattack_n_iter', 100))
            
            # Attack with full DIRE
            start_time = time.time()
            x_adv_full_sample = adversary_full.run_standard_evaluation(x_sample, y_sample, bs=1)
            attack_time = time.time() - start_time
            
            # Check perturbation
            perturbation_full = (x_adv_full_sample - x_sample).abs()
            pert_linf_full = perturbation_full.max().item()
            pert_l2_full = perturbation_full.norm().item()
            full_attack_times.append(float(attack_time))
            if not getattr(args, 'quiet_full', False):
                print(f"  Perturbation: L∞={pert_linf_full:.6f}, L2={pert_l2_full:.6f}")
                print(f"  Attack time: {attack_time:.2f}s")
                # Per-sample concise timing summary
                print(f"[time] sample#{sample_idx:04d} full_attack_time={attack_time:.2f}s")

            # Optionally save adversarial image (input space [0,1])
            if getattr(args, 'save_adv_images', False) and not getattr(args, 'quiet_full', False):
                base_name = f"sample_{global_sample_idx:06d}"
                # Save PNG for quick inspection
                save_path = os.path.join(save_dir_full, base_name + "_adv.png")
                save_image(x_adv_full_sample.clamp(0, 1).cpu(), save_path)
                print(f"  Saved full-mode adversarial image: {save_path}")

                # Additionally save tensors: clean, adv, delta, label
                delta = (x_adv_full_sample - x_sample).detach().cpu()
                payload = {
                    "clean": x_sample.detach().cpu(),
                    "adv": x_adv_full_sample.detach().cpu(),
                    "delta": delta,
                    "label": y_sample.detach().cpu(),
                }
                pt_path = os.path.join(save_dir_full, base_name + "_adv.pt")
                torch.save(payload, pt_path)
                print(f"  Saved full-mode adv tensors (clean/adv/delta/label): {pt_path}")
            
            if pert_linf_full < 1e-6:
                print(f"  [WARNING] No perturbation detected!")
            
            # Step 3: Evaluate adversarial example
            if not getattr(args, 'quiet_full', False):
                print("\n[Step 3] Evaluating adversarial example...")
            model_.reset_counter()
            model_.set_mode('full')
            model_.set_tag('eval_adv_full')
            
            with torch.no_grad():
                adv_pred_full = model(x_adv_full_sample)
                adv_logit_full = adv_pred_full.view(-1)
                adv_class_full = (adv_logit_full > 0).long()
                adv_correct_full = (adv_class_full == y_sample).item()
                full_adv_correct += adv_correct_full
                # update conditional counter: only count among those that were clean-correct
                if clean_correct:
                    full_adv_correct_on_clean += adv_correct_full
                # per-class updates
                if lbl == int(args.real_label):
                    full_adv_correct_real += adv_correct_full
                    if clean_correct:
                        full_adv_correct_on_clean_real += adv_correct_full
                elif lbl == int(args.fake_label):
                    full_adv_correct_fake += adv_correct_full
                    if clean_correct:
                        full_adv_correct_on_clean_fake += adv_correct_full
                
                if not getattr(args, 'quiet_full', False):
                    print(f"  Adversarial prediction: {adv_class_full.item()} (correct: {adv_correct_full})")
                    print(f"  Attack {'failed' if adv_correct_full else 'succeeded'}!")
            
            # Store full DIRE adversarial example
            all_x_adv_full.append(x_adv_full_sample.cpu())
        
        # ====================================================================
        # METHOD 3: DIRE-Space Attack (Attack classifier in DIRE output space)
        # ====================================================================
        if run_dire_space:
            print("\n" + "=" * 70)
            print("METHOD 3: DIRE-Space Attack (Attack in DIRE output, backproject to input)")
            print("=" * 70)
            
            # Step 1: Compute DIRE and gradient for this sample
            print("\n[Step 1] Computing DIRE and gradient...")
            model_.reset_counter()
            model_.set_tag('compute_gradient_dire_space')
            model_.set_mode('full')
            
            # Compute DIRE(x) and ∇DIRE(x) for this single sample
            dire_output, dire_gradient = model_.compute_dire_and_gradient(x_sample, batch_size=1)
            
            # Verify clean prediction in DIRE space (classifier-only)
            model_.set_mode('dire_space')
            with torch.no_grad():
                clean_pred = model(dire_output)
                clean_logit = clean_pred.view(-1)
                clean_class = (clean_logit > 0).long()
                clean_correct = (clean_class == y_sample).item()
                dire_space_clean_correct += clean_correct
                
                print(f"  Clean prediction: {clean_class.item()} (correct: {clean_correct})")

            # Also record clean prediction under FULL pipeline for conditional stage3 success
            model_.set_mode('full')
            with torch.no_grad():
                clean_pred_full = model(x_sample)
                clean_logit_full = clean_pred_full.view(-1)
                clean_class_full = (clean_logit_full > 0).long()
                clean_correct_full = (clean_class_full == y_sample).item()
                dire_space_full_clean_correct += clean_correct_full
                # per-class denom update for stage3 metrics
                lbl = int(y_sample.item())
                if lbl == int(args.real_label):
                    dire_stage3_total_real += 1
                    dire_space_full_clean_correct_real += clean_correct_full
                elif lbl == int(args.fake_label):
                    dire_stage3_total_fake += 1
                    dire_space_full_clean_correct_fake += clean_correct_full
            
            # Step 2: Attack classifier in DIRE output space
            print("\n[Step 2] Attacking classifier in DIRE output space...")
            model_.reset_counter()
            model_.set_mode('dire_space')
            model_.set_tag('attack_dire_space')

            # Gradient probe at DIRE space before attack
            _grad_probe(dire_output, y_sample, mode_name='dire_space')
            
            # Create AutoAttack for DIRE-space attack
            adversary_dire_space = AutoAttack(
                model_wrapped,
                norm=args.lp_norm,
                eps=args.adv_eps,  # Note: This eps is in DIRE output space!
                version=attack_version,
                attacks_to_run=attack_list,
                log_path=f'{log_dir}/log_sample_{sample_idx}_dire_space.txt',
                device=config.device,
                verbose=False
            )
            
            if attack_version == 'custom':
                adversary_dire_space.apgd.n_restarts = 1  # Single run for fair timing comparison
                adversary_dire_space.apgd.n_iter = int(getattr(args, 'autoattack_n_iter', 100))
            
            # Attack in DIRE space (optimize perturbations on dire_output)
            start_time = time.time()
            dire_adv = adversary_dire_space.run_standard_evaluation(dire_output, y_sample, bs=1)
            attack_time = time.time() - start_time
            
            # Compute perturbation in DIRE space
            delta_dire = (dire_adv - dire_output)
            pert_dire_linf = delta_dire.abs().max().item()
            pert_dire_l2 = delta_dire.norm().item()
            print(f"  DIRE-space perturbation: L∞={pert_dire_linf:.6f}, L2={pert_dire_l2:.6f}")
            print(f"  Attack time: {attack_time:.1f}s")
            
            # Stage 1: Evaluate classifier-only on DIRE space after attack
            print("\n[Stage 1] Classifier-only accuracy on perturbed DIRE map...")
            model_.reset_counter()
            model_.set_mode('dire_space')
            model_.set_tag('eval_adv_dire_space_stage1')
            with torch.no_grad():
                adv_pred_stage1 = model(dire_adv)
                adv_logit_stage1 = adv_pred_stage1.view(-1)
                adv_class_stage1 = (adv_logit_stage1 > 0).long()
                adv_correct_stage1 = (adv_class_stage1 == y_sample).item()
                dire_space_stage1_adv_correct += adv_correct_stage1
                print(f"  Stage1 prediction: {adv_class_stage1.item()} (correct: {adv_correct_stage1})")
                print(f"  Stage1 attack {'failed' if adv_correct_stage1 else 'succeeded'}!")
            
            # Step 3: Backproject to input space using matrix-free CG
            print("\n[Step 3] Backprojecting to input space with CG solver...")
            cg_start = time.time()
            
            # Define JVP and VJP functions for this sample
            # We need to recompute DIRE with grad enabled for JVP/VJP
            x_sample_for_jvp = x_sample.detach().requires_grad_(True)
            x_dire_for_jvp = (x_sample_for_jvp - 0.5) * 2.0
            
            # Recompute DIRE output with gradient tracking
            torch.manual_seed(int(args.seed) + sample_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed) + sample_idx)
            
            with torch.enable_grad():
                out_jvp = model_.runner.forward(
                    x_dire_for_jvp,
                    t_steps=int(args.t),
                    eot_reps=int(args.eot_reps),
                    fix_rand=bool(args.fix_rand)
                )
                dire_output_for_jvp = out_jvp['dire_map']
            
            def jvp_fn(v):
                """Compute J·v (Jacobian-vector product) at x_sample_for_jvp.
                Prefer exact JVP via autograd.functional.jvp; fallback to finite differences with clamping.
                """
                # Define f(x) that returns DIRE map in [0,1], with deterministic RNG
                def f_fn(x_in: torch.Tensor) -> torch.Tensor:
                    # Ensure determinism inside f
                    torch.manual_seed(int(args.seed) + sample_idx)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(args.seed) + sample_idx)
                    # [0,1] -> [-1,1]
                    x_dire_in = (x_in - 0.5) * 2.0
                    out_local = model_.runner.forward(
                        x_dire_in,
                        t_steps=int(args.t),
                        eot_reps=int(args.eot_reps),
                        fix_rand=bool(args.fix_rand)
                    )
                    return out_local['dire_map']

                # Try exact JVP
                try:
                    jv = torch.autograd.functional.jvp(
                        f_fn,
                        (x_sample_for_jvp,),
                        (v,),
                        strict=False
                    )[1]
                    # Sanitize NaNs/Infs
                    jv = torch.nan_to_num(jv, nan=0.0, posinf=0.0, neginf=0.0)
                    return jv
                except Exception:
                    # Fallback: finite difference with clamping and deterministic RNG
                    eps = 1e-4
                    x_plus = (x_sample_for_jvp + eps * v).clamp(0.0, 1.0)
                    # Reset RNG to keep same internal noise
                    torch.manual_seed(int(args.seed) + sample_idx)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(args.seed) + sample_idx)
                    with torch.no_grad():
                        out_plus = model_.runner.forward(
                            (x_plus - 0.5) * 2.0,
                            t_steps=int(args.t),
                            eot_reps=int(args.eot_reps),
                            fix_rand=bool(args.fix_rand)
                        )
                        dire_plus = out_plus['dire_map']
                    jv = (dire_plus - dire_output) / eps
                    jv = torch.nan_to_num(jv, nan=0.0, posinf=0.0, neginf=0.0)
                    return jv
            
            def vjp_fn(u):
                """Compute J^T·u using reverse-mode differentiation."""
                vjp = torch.autograd.grad(
                    outputs=dire_output_for_jvp,
                    inputs=x_sample_for_jvp,
                    grad_outputs=u,
                    retain_graph=True,
                    create_graph=False
                )[0]
                vjp = torch.nan_to_num(vjp, nan=0.0, posinf=0.0, neginf=0.0)
                return vjp
            
            # Compute RHS: J^T Δf
            rhs = vjp_fn(delta_dire)
            
            # Choose backprojection method
            mode = str(getattr(args, 'backproj_cache_mode', 'none'))
            cg_iters = int(getattr(args, 'backproj_cg_iters', 100))
            cg_lambda = float(getattr(args, 'backproj_cg_lambda', 1e-3))
            solver = str(getattr(args, 'backproj_solver', 'cg'))
            if mode == 'matrix':
                print(f"  Computing Jacobian directly for matrix solve (λ={cg_lambda:.1e})...")
                try:
                    # Compute device
                    compute_dev = x_sample_for_jvp.device
                    # Target device for matrix operations
                    target_dev_str = str(getattr(args, 'backproj_matrix_device', 'cpu') or 'cpu')
                    target_dev = torch.device(target_dev_str) if target_dev_str != 'auto' else compute_dev
                    
                    # Directly compute Jacobian J using jacfwd (MUCH faster than column-by-column)
                    J, unflatten_in, unflatten_out = compute_jacobian_direct(
                        model_runner=model_.runner,
                        x_sample=x_sample,
                        args=args,
                        sample_idx=global_sample_idx,
                        device=target_dev,
                        dtype=rhs.dtype,
                        verbose=True
                    )
                    
                    # Flatten delta_dire for matrix operations
                    delta_dire_flat = delta_dire.view(-1).to(device=target_dev, dtype=rhs.dtype)
                    
                    # Compute J^T @ delta_f
                    b = J.T @ delta_dire_flat
                    
                    # Compute J^T @ J + λI
                    print(f"  Computing J^T @ J + λI...")
                    t_jtj = time.time()
                    A = J.T @ J
                    n = A.shape[0]
                    A = A + cg_lambda * torch.eye(n, device=A.device, dtype=A.dtype)
                    jtj_time = time.time() - t_jtj
                    print(f"  J^T @ J computed in {jtj_time:.2f}s")
                    
                    # Solve A δ = b
                    t_solve = time.time()
                    try:
                        delta_flat = torch.linalg.solve(A, b)
                    except RuntimeError:
                        # Fallback to least-squares if singular/ill-conditioned
                        delta_flat = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
                    solve_time = time.time() - t_solve
                    print(f"  Linear system solved in {solve_time:.2f}s on {target_dev}")
                    
                    delta_input = unflatten_in(delta_flat.to(compute_dev))
                except Exception as e:
                    import traceback
                    print(f"  Matrix mode failed ({e}); falling back to CG.")
                    traceback.print_exc()
                    print(f"  Running CG: max_iters={cg_iters}, lambda={cg_lambda:.1e}")
                    delta_input = conjugate_gradient_solve(
                        jvp_fn=jvp_fn,
                        vjp_fn=vjp_fn,
                        rhs=rhs,
                        x0=None,
                        max_iters=cg_iters,
                        lambda_reg=cg_lambda,
                        tol=float(getattr(args, 'backproj_cg_tol', 1e-6))
                    )
            else:
                # Solve J δ ≈ Δf via LSQR or (J^T J + λI)δ = J^T Δf via CG
                if solver == 'lsqr':
                    print(f"  Running LSQR: max_iters={cg_iters} (no explicit damping)")
                    delta_input = lsqr_solve(
                        jvp_fn=jvp_fn,
                        vjp_fn=vjp_fn,
                        b=delta_dire,
                        x0=None,
                        max_iters=cg_iters,
                        tol=float(getattr(args, 'backproj_cg_tol', 1e-6))
                    )
                elif solver == 'gmres':
                    gmres_m = int(getattr(args, 'backproj_gmres_restart', 50))
                    print(f"  Running GMRES: restart={gmres_m}, max_iters={cg_iters}")
                    delta_input = gmres_solve(
                        jvp_fn=jvp_fn,
                        vjp_fn=vjp_fn,
                        b=delta_dire,
                        x0=None,
                        restart=gmres_m,
                        max_iters=cg_iters,
                        tol=float(getattr(args, 'backproj_cg_tol', 1e-6))
                    )
                else:
                    print(f"  Running CG: max_iters={cg_iters}, lambda={cg_lambda:.1e}")
                    delta_input = conjugate_gradient_solve(
                        jvp_fn=jvp_fn,
                        vjp_fn=vjp_fn,
                        rhs=rhs,
                        x0=None,
                        max_iters=cg_iters,
                        lambda_reg=cg_lambda,
                        tol=float(getattr(args, 'backproj_cg_tol', 1e-6))
                    )
            cg_time = time.time() - cg_start
            total_ds_time = float(attack_time + cg_time)
            dire_space_total_attack_times.append(total_ds_time)
            print(f"  CG time: {cg_time:.2f}s | Total attack time (DIRE-space + CG): {total_ds_time:.2f}s")
            # Per-sample concise timing summary
            print(f"[time] sample#{sample_idx:04d} dire_space_total_attack_time={total_ds_time:.2f}s (autoattack={attack_time:.2f}s, cg={cg_time:.2f}s)")
            # Sanitize solution and enforce epsilon budget and bounds
            delta_input = torch.nan_to_num(delta_input, nan=0.0, posinf=0.0, neginf=0.0)

            # Optional perceptual smoothing of backprojected delta (before budget clip)
            smooth_sigma = float(getattr(args, 'backproj_smooth_sigma', 0.0) or 0.0)
            smooth_ksize = int(getattr(args, 'backproj_smooth_ksize', 3) or 3)
            if smooth_sigma > 0.0:
                delta_input = _gaussian_smooth(delta_input, sigma=smooth_sigma, ksize=smooth_ksize)
            
            # Clip to eps budget in input space
            if args.lp_norm == 'Linf':
                delta_input = delta_input.clamp(-args.adv_eps, args.adv_eps)
            else:  # L2
                delta_norm = delta_input.norm()
                if delta_norm > args.adv_eps:
                    delta_input = delta_input * (args.adv_eps / delta_norm)
            
            # Generate adversarial example in input space
            x_adv_dire_space_sample = (x_sample + delta_input).clamp(0, 1)
            x_adv_dire_space_sample = torch.nan_to_num(x_adv_dire_space_sample, nan=0.0, posinf=1.0, neginf=0.0)
            
            pert_input_linf = delta_input.abs().max().item()
            pert_input_l2 = delta_input.norm().item()
            print(f"  Input-space perturbation: L∞={pert_input_linf:.6f}, L2={pert_input_l2:.6f}")

            # Optionally save adversarial image
            if getattr(args, 'save_adv_images', False):
                save_path = os.path.join(save_dir_dire_space, f'sample_{global_sample_idx:06d}_adv.png')
                save_image(x_adv_dire_space_sample.clamp(0,1).cpu(), save_path)
                print(f"  Saved dire-space (backprojected) adversarial image: {save_path}")
            
            
            # Stage 2: Removed (linearized pipeline deprecated)
            # Directly jump to Stage 3 for full pipeline evaluation
            
            # Step 4 / Stage 3: Evaluate on full DIRE
            if not getattr(args, 'quiet_full', False):
                print("\n[Step 4] Evaluating on full DIRE...")
            model_.reset_counter()
            model_.set_mode('full')
            model_.set_tag('eval_adv_dire_space')
            
            with torch.no_grad():
                adv_pred_dire_space = model(x_adv_dire_space_sample)
                adv_logit_dire_space = adv_pred_dire_space.view(-1)
                adv_class_dire_space = (adv_logit_dire_space > 0).long()
                adv_correct_dire_space = (adv_class_dire_space == y_sample).item()
                dire_space_stage3_adv_correct += adv_correct_dire_space
                # conditional on full clean-correct subset
                if clean_correct_full:
                    dire_space_stage3_adv_correct_on_cleanfull += adv_correct_dire_space
                # per-class updates
                if lbl == int(args.real_label):
                    dire_space_stage3_adv_correct_real += adv_correct_dire_space
                    if clean_correct_full:
                        dire_space_stage3_adv_correct_on_cleanfull_real += adv_correct_dire_space
                elif lbl == int(args.fake_label):
                    dire_space_stage3_adv_correct_fake += adv_correct_dire_space
                    if clean_correct_full:
                        dire_space_stage3_adv_correct_on_cleanfull_fake += adv_correct_dire_space
                
                print(f"  Stage3 (full) prediction: {adv_class_dire_space.item()} (correct: {adv_correct_dire_space})")
                print(f"  Stage3 attack {'failed' if adv_correct_dire_space else 'succeeded'}!")
            
            # Store DIRE-space adversarial example
            all_x_adv_dire_space.append(x_adv_dire_space_sample.cpu())

        # ====================================================================
        # METHOD 4: Zeroth-Order Black-Box Attack (Input Space)
        # ====================================================================
        if run_zeroth:
            print("\n" + "=" * 70)
            print("METHOD 4: Zeroth-Order Black-Box (Input-space NES/SPSA-style)")
            print("=" * 70)

            # Step 1: Clean prediction (full pipeline)
            print("\n[Step 1] Checking clean prediction (full pipeline)...")
            model_.reset_counter()
            model_.set_mode('full')
            model_.set_tag('clean_zeroth')
            with torch.no_grad():
                clean_pred = model(x_sample)
                clean_logit = clean_pred.view(-1)
                clean_class = (clean_logit > 0).long()
                clean_correct = (clean_class == y_sample).item()
                zeroth_clean_correct += clean_correct
                # per-class
                lbl = int(y_sample.item())
                if lbl == int(args.real_label):
                    zeroth_total_real += 1
                    zeroth_clean_correct_real += clean_correct
                elif lbl == int(args.fake_label):
                    zeroth_total_fake += 1
                    zeroth_clean_correct_fake += clean_correct
                print(f"  Clean prediction: {clean_class.item()} (correct: {clean_correct})")

            # Step 2: Zeroth-order optimization in input space
            print("\n[Step 2] Running zeroth-order attack in input space...")
            start_time = time.time()
            x0 = x_sample.detach()
            x_adv = x0.clone()
            eps = float(args.adv_eps)
            norm = args.lp_norm
            iters = int(getattr(args, 'zo_iters', 100))
            q = int(getattr(args, 'zo_q', 64))
            q_batch = int(getattr(args, 'zo_batch_q', q))
            zo_obj = str(getattr(args, 'zo_objective', 'ce'))
            mu = float(getattr(args, 'zo_mu', 0.01))
            alpha = float(getattr(args, 'zo_alpha', 0.01))
            symmetric_fd = bool(getattr(args, 'zo_symmetric', True))
            interior_bias = float(getattr(args, 'zo_interior_bias', 0.0))
            query_clip_eps = float(getattr(args, 'zo_query_clip_eps', 0.0))
            # Early stopping config
            es_on_success = bool(getattr(args, 'zo_early_stop_on_success', False))
            es_patience = int(getattr(args, 'zo_early_patience', 10))
            es_min_delta = float(getattr(args, 'zo_early_min_delta', 1e-6))
            best_loss = -float('inf')
            no_improve = 0

            def _compute_losses_from_logits(logits2: torch.Tensor, y: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
                """
                Compute zeroth objective losses from two-class logits.
                - ce:      standard cross-entropy on true label (maximize)
                - margin:  max_other - logit_true (maximize)
                - dlr:     (max_other - logit_true) / (top - bottom + eps) (maximize)
                Returns per-sample losses unless reduction='mean'.
                """
                if logits2.dim() != 2 or logits2.size(1) != 2:
                    print(f"  [zeroth] unexpected logits shape: {tuple(logits2.shape)} (expect [B,2])")
                if zo_obj == 'ce':
                    loss_vec = F.cross_entropy(logits2, y, reduction='none')
                else:
                    y = y.view(-1)
                    idx_other = 1 - y
                    true_logit = logits2[torch.arange(logits2.size(0), device=logits2.device), y]
                    other_logit = logits2[torch.arange(logits2.size(0), device=logits2.device), idx_other]
                    margin = other_logit - true_logit  # larger => worse for true class
                    if zo_obj == 'margin':
                        loss_vec = margin
                    else:  # dlr
                        top, _ = logits2.max(dim=1)
                        bottom, _ = logits2.min(dim=1)
                        denom = (top - bottom).abs() + 1e-6
                        loss_vec = margin / denom
                if reduction == 'mean':
                    return loss_vec.mean()
                return loss_vec

            def _loss_fn(x_in: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    logits2 = model_wrapped(x_in)
                    loss = _compute_losses_from_logits(logits2, y_sample, reduction='mean')
                return loss

            # Project helper
            def _project(x_adv_curr: torch.Tensor) -> torch.Tensor:
                delta = x_adv_curr - x0
                if norm == 'Linf':
                    delta = delta.clamp(-eps, eps)
                else:  # L2
                    d = delta.view(delta.size(0), -1)
                    d_norm = d.norm(p=2, dim=1, keepdim=True) + 1e-12
                    factor = torch.minimum(torch.ones_like(d_norm), eps / d_norm)
                    d = d * factor
                    delta = d.view_as(delta)
                x_proj = (x0 + delta).clamp(0.0, 1.0)
                return x_proj

            # NES/SPSA-style gradient estimation loop (vectorized over q directions with mini-batches)
            B, C, H, W = x0.shape
            # Adaptive mu to avoid vanishing finite-difference signals
            adapt_mu = bool(getattr(args, 'zo_adapt_mu', True))
            adapt_mu_factor = float(getattr(args, 'zo_adapt_mu_factor', 2.0))
            adapt_mu_min_diff = float(getattr(args, 'zo_adapt_mu_min_diff', 1e-4))
            adapt_mu_max = float(getattr(args, 'zo_adapt_mu_max', min(0.1, eps)))

            for it in range(iters):
                # Accumulate gradient estimate
                grad_est = torch.zeros_like(x0)
                rem = q
                fdiff_abs_accum = 0.0
                fdiff_batches = 0
                # One-sided base loss (computed once per iteration if needed)
                if not symmetric_fd:
                    with torch.no_grad():
                        logits_base = model_wrapped(x_adv)
                        base_loss = _compute_losses_from_logits(logits_base, y_sample, reduction='mean')
                while rem > 0:
                    qb = min(q_batch, rem)
                    # Sample qb directions
                    if norm == 'Linf':
                        U = torch.empty((qb, C, H, W), device=x0.device, dtype=x0.dtype).uniform_(-1.0, 1.0).sign()
                    else:  # L2 directions
                        U = torch.randn((qb, C, H, W), device=x0.device, dtype=x0.dtype)
                        U = U / (U.view(qb, -1).norm(p=2, dim=1).view(qb, 1, 1, 1) + 1e-12)
                    # Build positive/negative query batches
                    X_base = x_adv.expand(qb, -1, -1, -1)
                    if interior_bias > 0.0:
                        # Shift inward from bounds for query construction only
                        X_base = X_base * (1.0 - 2.0 * interior_bias) + interior_bias
                    if symmetric_fd:
                        X_pos = X_base + mu * U
                        X_neg = X_base - mu * U
                        if query_clip_eps > 0.0:
                            X_pos = X_pos.clamp(query_clip_eps, 1.0 - query_clip_eps)
                            X_neg = X_neg.clamp(query_clip_eps, 1.0 - query_clip_eps)
                        else:
                            X_pos = X_pos.clamp(0.0, 1.0)
                            X_neg = X_neg.clamp(0.0, 1.0)
                    else:
                        X_pos = X_base + mu * U
                        if query_clip_eps > 0.0:
                            X_pos = X_pos.clamp(query_clip_eps, 1.0 - query_clip_eps)
                        else:
                            X_pos = X_pos.clamp(0.0, 1.0)
                    # Evaluate losses for both in one forward (no gradients)
                    with torch.no_grad():
                        if symmetric_fd:
                            logits_concat = model_wrapped(torch.cat([X_pos, X_neg], dim=0))
                            # Expect [2*qb, 2]
                            yb = y_sample.repeat(2 * qb)
                            losses = _compute_losses_from_logits(logits_concat, yb, reduction='none')
                            f_pos = losses[:qb]
                            f_neg = losses[qb:]
                            fdiff_vec = (f_pos - f_neg)
                            fdiff = (fdiff_vec / (2.0 * mu)).view(qb, 1, 1, 1)
                            # Track finite-difference signal magnitude
                            fdiff_abs_accum += fdiff_vec.abs().mean().item()
                        else:
                            logits_pos = model_wrapped(X_pos)
                            yb = y_sample.repeat(qb)
                            f_pos = _compute_losses_from_logits(logits_pos, yb, reduction='none')
                            fdiff_vec = (f_pos - base_loss)
                            fdiff = (fdiff_vec / mu).view(qb, 1, 1, 1)
                            fdiff_abs_accum += fdiff_vec.abs().mean().item()
                        # Sum contribution over the mini-batch
                        grad_est += (fdiff * U).sum(dim=0, keepdim=True)
                        fdiff_batches += 1
                    rem -= qb
                grad_est /= float(q)

                # If signal is too small for several batches, consider increasing mu
                if adapt_mu and fdiff_batches > 0:
                    fdiff_abs_mean = fdiff_abs_accum / max(1, fdiff_batches)
                    if fdiff_abs_mean < adapt_mu_min_diff and mu < adapt_mu_max:
                        new_mu = min(mu * adapt_mu_factor, adapt_mu_max)
                        if new_mu > mu:
                            print(f"    [adapt-mu] |f_pos-f_neg| mean {fdiff_abs_mean:.2e} < {adapt_mu_min_diff:.2e}; increasing mu {mu:.4f} -> {new_mu:.4f}")
                            mu = new_mu

                # Update
                if norm == 'Linf':
                    x_adv = x_adv + alpha * grad_est.sign()
                else:  # L2
                    g = grad_est.view(B, -1)
                    g_norm = g.norm(p=2, dim=1, keepdim=True) + 1e-12
                    step = (alpha * (g / g_norm)).view_as(x_adv)
                    x_adv = x_adv + step

                # Project to epsilon-ball and image bounds
                x_adv = _project(x_adv)

                # Compute current loss and (optionally) early stop
                with torch.no_grad():
                    logits_cur = model_wrapped(x_adv)
                    cur_loss = F.cross_entropy(logits_cur, y_sample)
                    # Compute margin (the actual attack objective)
                    obj_loss_val = _compute_losses_from_logits(logits_cur, y_sample, reduction='mean')
                    # logits_cur expected shape [B,2]; take argmax over class dim
                    pred_cls = logits_cur.argmax(dim=1)
                    corr_flag = (pred_cls == y_sample).item()

                # Progress print every ~20% iterations
                if (it + 1) % max(1, iters // 5) == 0:
                    grad_norm = grad_est.abs().mean().item()
                    fdiff_mean = fdiff_abs_accum / max(1, fdiff_batches) if fdiff_batches > 0 else 0.0
                    print(f"    iter {it+1}/{iters} | correct={corr_flag} | ce_loss={cur_loss.item():.4f} | {zo_obj}_obj={obj_loss_val.item():.4f} | grad_norm={grad_norm:.4e} | fdiff={fdiff_mean:.4e}")

                # Early stop on success
                if es_on_success and not corr_flag:
                    print(f"    [early-stop] attack already successful at iter {it+1}; stopping.")
                    break

                # Early stop on loss plateau (maximize the attack objective, not CE)
                if obj_loss_val.item() > best_loss + es_min_delta:
                    best_loss = obj_loss_val.item()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= es_patience:
                        print(f"    [early-stop] no improvement for {es_patience} iters; stopping.")
                        break

            atk_time = time.time() - start_time
            zeroth_attack_times.append(float(atk_time))
            print(f"  Zeroth-order attack time: {atk_time:.2f}s")
            print(f"[time] sample#{sample_idx:04d} zeroth_attack_time={atk_time:.2f}s")

            # Evaluate
            model_.reset_counter()
            model_.set_mode('full')
            model_.set_tag('eval_adv_zeroth')
            with torch.no_grad():
                adv_pred = model(x_adv)
                adv_logit = adv_pred.view(-1)
                adv_class = (adv_logit > 0).long()
                adv_correct = (adv_class == y_sample).item()
                zeroth_adv_correct += adv_correct
                if clean_correct:
                    zeroth_adv_correct_on_clean += adv_correct
                if lbl == int(args.real_label):
                    zeroth_adv_correct_real += adv_correct
                    if clean_correct:
                        zeroth_adv_correct_on_clean_real += adv_correct
                elif lbl == int(args.fake_label):
                    zeroth_adv_correct_fake += adv_correct
                    if clean_correct:
                        zeroth_adv_correct_on_clean_fake += adv_correct
                print(f"  Adversarial prediction: {adv_class.item()} (correct: {adv_correct})")
                print(f"  Attack {'failed' if adv_correct else 'succeeded'}!")

            # Save image
            if getattr(args, 'save_adv_images', False):
                save_path = os.path.join(save_dir_zeroth, f'sample_{global_sample_idx:06d}_adv.png')
                save_image(x_adv.clamp(0,1).cpu(), save_path)
                print(f"  Saved zeroth-order adversarial image: {save_path}")

            # Store
            all_x_adv_zeroth.append(x_adv.cpu())
        
        # ====================================================================
        # Progress Summary
        # ====================================================================
        print(f"\n{'=' * 70}")
        print(f"Progress after {global_sample_idx + 1} samples (batch: {sample_idx + 1}/{total_samples}):")
        if run_linear:
            # Use cumulative numerators (prefix from previous batches + current batch counters)
            linear_clean_acc = (pp_linear_clean + linear_clean_correct) / (global_sample_idx + 1)
            linear_robust_acc = (pp_linear_adv + linear_adv_correct) / (global_sample_idx + 1)
            print(f"  METHOD 1 (Linear):     Clean {linear_clean_acc:.2%}, Robust {linear_robust_acc:.2%}")
        if run_full:
            full_clean_acc = (pp_full_clean + full_clean_correct) / (global_sample_idx + 1)
            full_robust_acc = (pp_full_adv + full_adv_correct) / (global_sample_idx + 1)
            full_overall_success = 1 - full_robust_acc
            full_cond_success = (1 - ((pp_full_on_clean + full_adv_correct_on_clean) / (pp_full_clean + full_clean_correct))) if (pp_full_clean + full_clean_correct) > 0 else 0.0
            print(f"  METHOD 2 (Full):       Clean {full_clean_acc:.2%}, Robust {full_robust_acc:.2%}")
            print(f"                         Success(overall) {full_overall_success:.2%}, Success(on clean-only) {full_cond_success:.2%}")
            # Per-class running success (guard zeros)
            if (pp_full_total_real + pp_full_total_fake + full_total_real + full_total_fake) > 0:
                tot_real = pp_full_total_real + full_total_real
                tot_fake = pp_full_total_fake + full_total_fake
                if tot_real > 0:
                    real_overall_succ = 1 - ((pp_full_adv_real + full_adv_correct_real) / tot_real)
                    denom_clean_real = pp_full_clean_real + full_clean_correct_real
                    real_clean_cond_succ = (1 - ((pp_full_on_clean_real + full_adv_correct_on_clean_real) / max(denom_clean_real, 1))) if denom_clean_real > 0 else 0.0
                    print(f"                         [by-class] real Success(overall) {real_overall_succ:.2%}, Success(on clean-only) {real_clean_cond_succ:.2%}")
                if tot_fake > 0:
                    fake_overall_succ = 1 - ((pp_full_adv_fake + full_adv_correct_fake) / tot_fake)
                    denom_clean_fake = pp_full_clean_fake + full_clean_correct_fake
                    fake_clean_cond_succ = (1 - ((pp_full_on_clean_fake + full_adv_correct_on_clean_fake) / max(denom_clean_fake, 1))) if denom_clean_fake > 0 else 0.0
                    print(f"                         [by-class] fake Success(overall) {fake_overall_succ:.2%}, Success(on clean-only) {fake_clean_cond_succ:.2%}")
        if run_dire_space:
            dire_space_clean_acc = (pp_ds_clean + dire_space_clean_correct) / (global_sample_idx + 1)
            stage1_acc = (pp_ds_stage1 + dire_space_stage1_adv_correct) / (global_sample_idx + 1)
            stage3_acc = (pp_ds_stage3 + dire_space_stage3_adv_correct) / (global_sample_idx + 1)
            stage3_overall_success = 1 - stage3_acc
            denom_full_clean = pp_ds_full_clean + dire_space_full_clean_correct
            num_on_clean = pp_ds_on_cleanfull + dire_space_stage3_adv_correct_on_cleanfull
            stage3_cond_success = (1 - (num_on_clean / denom_full_clean)) if denom_full_clean > 0 else 0.0
            print(f"  METHOD 3 (DIRE-Space): Clean {dire_space_clean_acc:.2%}, Stage1 {stage1_acc:.2%}, Stage3 {stage3_acc:.2%}")
            print(f"                         Stage3 Success(overall) {stage3_overall_success:.2%}, Stage3 Success(on clean-only) {stage3_cond_success:.2%}")
            # Per-class running success for Stage3
            if (pp_ds_total_real + pp_ds_total_fake + dire_stage3_total_real + dire_stage3_total_fake) > 0:
                tot_real = pp_ds_total_real + dire_stage3_total_real
                tot_fake = pp_ds_total_fake + dire_stage3_total_fake
                if tot_real > 0:
                    real_stage3_overall_succ = 1 - ((pp_ds_stage3_real + dire_space_stage3_adv_correct_real) / tot_real)
                    denom_real = pp_ds_full_clean_real + dire_space_full_clean_correct_real
                    real_stage3_cond_succ = (1 - ((pp_ds_on_cleanfull_real + dire_space_stage3_adv_correct_on_cleanfull_real) / max(denom_real, 1))) if denom_real > 0 else 0.0
                    print(f"                         [by-class] real Stage3 Success(overall) {real_stage3_overall_succ:.2%}, Stage3 Success(on clean-only) {real_stage3_cond_succ:.2%}")
                if tot_fake > 0:
                    fake_stage3_overall_succ = 1 - ((pp_ds_stage3_fake + dire_space_stage3_adv_correct_fake) / tot_fake)
                    denom_fake = pp_ds_full_clean_fake + dire_space_full_clean_correct_fake
                    fake_stage3_cond_succ = (1 - ((pp_ds_on_cleanfull_fake + dire_space_stage3_adv_correct_on_cleanfull_fake) / max(denom_fake, 1))) if denom_fake > 0 else 0.0
                    print(f"                         [by-class] fake Stage3 Success(overall) {fake_stage3_overall_succ:.2%}, Stage3 Success(on clean-only) {fake_stage3_cond_succ:.2%}")
        if run_zeroth:
            zero_clean_acc = (pp_z_clean + zeroth_clean_correct) / (global_sample_idx + 1)
            zero_robust_acc = (pp_z_adv + zeroth_adv_correct) / (global_sample_idx + 1)
            zero_overall_success = 1 - zero_robust_acc
            zero_cond_success = (1 - ((pp_z_on_clean + zeroth_adv_correct_on_clean) / (pp_z_clean + zeroth_clean_correct))) if (pp_z_clean + zeroth_clean_correct) > 0 else 0.0
            print(f"  METHOD 4 (Zeroth):     Clean {zero_clean_acc:.2%}, Robust {zero_robust_acc:.2%}")
            print(f"                         Success(overall) {zero_overall_success:.2%}, Success(on clean-only) {zero_cond_success:.2%}")
            if (pp_z_total_real + pp_z_total_fake + zeroth_total_real + zeroth_total_fake) > 0:
                tot_real = pp_z_total_real + zeroth_total_real
                tot_fake = pp_z_total_fake + zeroth_total_fake
                if tot_real > 0:
                    real_overall_succ = 1 - ((pp_z_adv_real + zeroth_adv_correct_real) / tot_real)
                    denom_clean_real = pp_z_clean_real + zeroth_clean_correct_real
                    real_clean_cond_succ = (1 - ((pp_z_on_clean_real + zeroth_adv_correct_on_clean_real) / max(denom_clean_real, 1))) if denom_clean_real > 0 else 0.0
                    print(f"                         [by-class] real Success(overall) {real_overall_succ:.2%}, Success(on clean-only) {real_clean_cond_succ:.2%}")
                if tot_fake > 0:
                    fake_overall_succ = 1 - ((pp_z_adv_fake + zeroth_adv_correct_fake) / tot_fake)
                    denom_clean_fake = pp_z_clean_fake + zeroth_clean_correct_fake
                    fake_clean_cond_succ = (1 - ((pp_z_on_clean_fake + zeroth_adv_correct_on_clean_fake) / max(denom_clean_fake, 1))) if denom_clean_fake > 0 else 0.0
                    print(f"                         [by-class] fake Success(overall) {fake_overall_succ:.2%}, Success(on clean-only) {fake_clean_cond_succ:.2%}")
        print(f"{'=' * 70}")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # ========================================================================
    # Final Results
    # ========================================================================
    if not suppress_final_print:
        print(f"\n{'=' * 70}")
        if run_linear and run_full:
            print("FINAL RESULTS - METHOD COMPARISON")
        elif run_linear:
            print("FINAL RESULTS - METHOD 1: Linearized Attack")
        elif run_full:
            print("FINAL RESULTS - METHOD 2: Full DIRE Attack")
        elif run_dire_space:
            print("FINAL RESULTS - METHOD 3: DIRE-Space Attack")
        elif run_zeroth:
            print("FINAL RESULTS - METHOD 4: Zeroth-Order Black-Box Attack")
        else:
            print("FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Total samples evaluated: {total_samples}")
        print()
    
    # METHOD 1: Linearized Attack
    if run_linear:
        linear_clean_acc = linear_clean_correct / total_samples
        linear_robust_acc = linear_adv_correct / total_samples
        linear_overall_success = 1 - linear_robust_acc
        linear_attack_success = (1 - linear_robust_acc/linear_clean_acc) if linear_clean_acc > 0 else 0
        
        if not suppress_final_print:
            print("METHOD 1: Linearized DIRE Attack (Fast)")
            print(f"  Clean accuracy:   {linear_clean_acc:.2%} ({linear_clean_correct}/{total_samples})")
            print(f"  Robust accuracy:  {linear_robust_acc:.2%} ({linear_adv_correct}/{total_samples})")
            print(f"  Attack success (overall):        {linear_overall_success:.2%}")
            print(f"  Attack success (on clean-only):  {linear_attack_success:.2%}")
            print()
    
    # METHOD 2: Full DIRE Attack (compute metrics regardless of quiet_full for downstream summary/comparison)
    if run_full:
        full_clean_acc = full_clean_correct / total_samples
        full_robust_acc = full_adv_correct / total_samples
        # Overall success on all samples
        full_overall_attack_success = 1 - full_robust_acc
        # Conditional success on clean-correct subset
        full_cond_success = (1 - (full_adv_correct_on_clean / full_clean_correct)) if full_clean_correct > 0 else 0

    if run_full and not getattr(args, 'quiet_full', False):
        
        if not suppress_final_print:
            print("METHOD 2: Full DIRE Attack (Slow but Accurate)")
            print(f"  Clean accuracy:   {full_clean_acc:.2%} ({full_clean_correct}/{total_samples})")
            print(f"  Robust accuracy:  {full_robust_acc:.2%} ({full_adv_correct}/{total_samples})")
            print(f"  Attack success (overall):        {full_overall_attack_success:.2%}")
            print(f"  Attack success (on clean-only):  {full_cond_success:.2%}")
            # Per-class final summary
            if (full_total_real + full_total_fake) > 0:
                if full_total_real > 0:
                    real_overall_succ = 1 - (full_adv_correct_real / max(full_total_real, 1))
                    real_clean_cond_succ = (1 - (full_adv_correct_on_clean_real / max(full_clean_correct_real, 1))) if full_clean_correct_real > 0 else 0.0
                    print(f"  [by-class] real Success(overall): {real_overall_succ:.2%}  | Success(on clean-only): {real_clean_cond_succ:.2%}")
                if full_total_fake > 0:
                    fake_overall_succ = 1 - (full_adv_correct_fake / max(full_total_fake, 1))
                    fake_clean_cond_succ = (1 - (full_adv_correct_on_clean_fake / max(full_clean_correct_fake, 1))) if full_clean_correct_fake > 0 else 0.0
                    print(f"  [by-class] fake Success(overall): {fake_overall_succ:.2%}  | Success(on clean-only): {fake_clean_cond_succ:.2%}")
            print()
    
    # METHOD 3: DIRE-Space Attack
    if run_dire_space:
        dire_space_clean_acc = dire_space_clean_correct / total_samples
        stage1_acc = dire_space_stage1_adv_correct / total_samples
        stage3_acc = dire_space_stage3_adv_correct / total_samples
        # Overall stage3 success on all samples (evaluated with full pipeline)
        stage3_overall_success = 1 - stage3_acc
        # Conditional stage3 success on samples that the FULL pipeline gets correct
        stage3_cond_success = (1 - (dire_space_stage3_adv_correct_on_cleanfull / dire_space_full_clean_correct)) if dire_space_full_clean_correct > 0 else 0
        
        if not suppress_final_print:
            print("METHOD 3: DIRE-Space Attack (CG backprojection from DIRE output)")
            print(f"  Clean accuracy:           {dire_space_clean_acc:.2%} ({dire_space_clean_correct}/{total_samples})")
            print(f"  Stage1 robust (DIRE map): {stage1_acc:.2%} ({dire_space_stage1_adv_correct}/{total_samples})")
            print(f"  Stage3 robust (full):     {stage3_acc:.2%} ({dire_space_stage3_adv_correct}/{total_samples})")
            print(f"  Stage3 success (overall): {stage3_overall_success:.2%}")
            print(f"  Stage3 success (on clean-only, full-pipeline clean): {stage3_cond_success:.2%}")
            # Per-class final summary for Stage3
            if (dire_stage3_total_real + dire_stage3_total_fake) > 0:
                if dire_stage3_total_real > 0:
                    real_stage3_overall_succ = 1 - (dire_space_stage3_adv_correct_real / max(dire_stage3_total_real, 1))
                    real_stage3_cond_succ = (1 - (dire_space_stage3_adv_correct_on_cleanfull_real / max(dire_space_full_clean_correct_real, 1))) if dire_space_full_clean_correct_real > 0 else 0.0
                    print(f"  [by-class] real Stage3 Success(overall): {real_stage3_overall_succ:.2%}  | Stage3 Success(on clean-only): {real_stage3_cond_succ:.2%}")
                if dire_stage3_total_fake > 0:
                    fake_stage3_overall_succ = 1 - (dire_space_stage3_adv_correct_fake / max(dire_stage3_total_fake, 1))
                    fake_stage3_cond_succ = (1 - (dire_space_stage3_adv_correct_on_cleanfull_fake / max(dire_space_full_clean_correct_fake, 1))) if dire_space_full_clean_correct_fake > 0 else 0.0
                    print(f"  [by-class] fake Stage3 Success(overall): {fake_stage3_overall_succ:.2%}  | Stage3 Success(on clean-only): {fake_stage3_cond_succ:.2%}")
            print()

    # METHOD 4: Zeroth-Order Black-Box Attack
    if run_zeroth:
        zero_clean_acc = zeroth_clean_correct / total_samples
        zero_robust_acc = zeroth_adv_correct / total_samples
        zero_overall_success = 1 - zero_robust_acc
        zero_cond_success = (1 - (zeroth_adv_correct_on_clean / zeroth_clean_correct)) if zeroth_clean_correct > 0 else 0
        if not suppress_final_print:
            print("METHOD 4: Zeroth-Order Black-Box (Input-space)")
            print(f"  Clean accuracy:   {zero_clean_acc:.2%} ({zeroth_clean_correct}/{total_samples})")
            print(f"  Robust accuracy:  {zero_robust_acc:.2%} ({zeroth_adv_correct}/{total_samples})")
            print(f"  Attack success (overall):        {zero_overall_success:.2%}")
            print(f"  Attack success (on clean-only):  {zero_cond_success:.2%}")
            # Per-class final summary
            if (zeroth_total_real + zeroth_total_fake) > 0:
                if zeroth_total_real > 0:
                    real_overall_succ = 1 - (zeroth_adv_correct_real / max(zeroth_total_real, 1))
                    real_clean_cond_succ = (1 - (zeroth_adv_correct_on_clean_real / max(zeroth_clean_correct_real, 1))) if zeroth_clean_correct_real > 0 else 0.0
                    print(f"  [by-class] real Success(overall): {real_overall_succ:.2%}  | Success(on clean-only): {real_clean_cond_succ:.2%}")
                if zeroth_total_fake > 0:
                    fake_overall_succ = 1 - (zeroth_adv_correct_fake / max(zeroth_total_fake, 1))
                    fake_clean_cond_succ = (1 - (zeroth_adv_correct_on_clean_fake / max(zeroth_clean_correct_fake, 1))) if zeroth_clean_correct_fake > 0 else 0.0
                    print(f"  [by-class] fake Success(overall): {fake_overall_succ:.2%}  | Success(on clean-only): {fake_clean_cond_succ:.2%}")
            print()
    
    # Comparison (only if both methods were run)
    if run_linear and run_full and not suppress_final_print:
        print("COMPARISON:")
        print(f"  Attack success difference (overall): {abs(full_overall_attack_success - (1 - linear_robust_acc)):.2%}")
        print(f"  {'Full DIRE attack is more effective' if full_overall_attack_success > (1 - linear_robust_acc) else 'Linearization is sufficient'}")
    # Print compact per-sample attack time summary after all samples
    if not suppress_final_print:
        if len(full_attack_times) > 0:
            print("Per-sample attack times [FULL DIRE] (s):")
            print(full_attack_times)
        if len(dire_space_total_attack_times) > 0:
            print("Per-sample attack times [DIRE-Space total] (s):")
            print(dire_space_total_attack_times)
        if len(zeroth_attack_times) > 0:
            print("Per-sample attack times [Zeroth] (s):")
            print(zeroth_attack_times)
    
    if not suppress_final_print:
        print(f"{'=' * 70}\n")
    
    # Concatenate and save adversarial examples
    if (not suppress_save) and run_linear and len(all_x_adv_linear) > 0:
        x_adv_dire_linear = torch.cat(all_x_adv_linear, dim=0)
        torch.save([x_adv_dire_linear, y_val], f'{log_dir}/x_adv_dire_linear_sd{args.seed}.pt')
        if not suppress_final_print:
            print(f"Saved linear attack: {log_dir}/x_adv_dire_linear_sd{args.seed}.pt")
    
    if (not suppress_save) and run_full and len(all_x_adv_full) > 0:
        x_adv_dire_full = torch.cat(all_x_adv_full, dim=0)
        torch.save([x_adv_dire_full, y_val], f'{log_dir}/x_adv_dire_full_sd{args.seed}.pt')
        if not suppress_final_print:
            print(f"Saved full attack:       {log_dir}/x_adv_dire_full_sd{args.seed}.pt")
    
    if (not suppress_save) and run_dire_space and len(all_x_adv_dire_space) > 0:
        x_adv_dire_space = torch.cat(all_x_adv_dire_space, dim=0)
        torch.save([x_adv_dire_space, y_val], f'{log_dir}/x_adv_dire_space_sd{args.seed}.pt')
        if not suppress_final_print:
            print(f"Saved DIRE-space attack: {log_dir}/x_adv_dire_space_sd{args.seed}.pt")

    if (not suppress_save) and run_zeroth and len(all_x_adv_zeroth) > 0:
        x_adv_zeroth = torch.cat(all_x_adv_zeroth, dim=0)
        torch.save([x_adv_zeroth, y_val], f'{log_dir}/x_adv_zeroth_sd{args.seed}.pt')
        if not suppress_final_print:
            print(f"Saved zeroth-order attack: {log_dir}/x_adv_zeroth_sd{args.seed}.pt")
    
    # Save summary
    summary = {
        'attack_version': attack_version,
        'attack_list': attack_list,
        'attack_method': args.attack_method,
        'lp_norm': args.lp_norm,
        'epsilon': args.adv_eps,
        'total_samples': total_samples,
    }
    
    # Add METHOD 1 results if run
    if run_linear:
        summary.update({
            'linear_clean_accuracy': linear_clean_acc,
            'linear_robust_accuracy': linear_robust_acc,
            'linear_attack_success_overall': linear_overall_success,
            'linear_attack_success_on_clean': linear_attack_success,
        })
    
    # Add METHOD 2 results if run
    if run_full:
        summary.update({
            'full_clean_accuracy': full_clean_acc,
            'full_robust_accuracy': full_robust_acc,
            'full_attack_success_overall': full_overall_attack_success,
            'full_attack_success_on_clean': full_cond_success,
            'full_clean_correct_count': full_clean_correct,
            'full_adv_correct_on_clean_count': full_adv_correct_on_clean,
            # per-class counts
            'full_total_real': full_total_real,
            'full_total_fake': full_total_fake,
            'full_clean_correct_real': full_clean_correct_real,
            'full_clean_correct_fake': full_clean_correct_fake,
            'full_adv_correct_real': full_adv_correct_real,
            'full_adv_correct_fake': full_adv_correct_fake,
            'full_adv_correct_on_clean_real': full_adv_correct_on_clean_real,
            'full_adv_correct_on_clean_fake': full_adv_correct_on_clean_fake,
            # timing for streaming aggregate
            'full_attack_time_sum': float(sum(full_attack_times)) if len(full_attack_times) > 0 else 0.0,
            'full_attack_time_count': int(len(full_attack_times)),
        })
    
    # Add METHOD 3 results if run
    if run_dire_space:
        summary.update({
            'dire_space_clean_accuracy': dire_space_clean_acc,
            'dire_space_stage1_robust_accuracy': stage1_acc,
            'dire_space_stage3_robust_accuracy': stage3_acc,
            'dire_space_stage3_success_overall': stage3_overall_success,
            'dire_space_stage3_success_on_clean_full': stage3_cond_success,
            'dire_space_full_clean_correct_count': dire_space_full_clean_correct,
            'dire_space_stage3_adv_correct_on_cleanfull_count': dire_space_stage3_adv_correct_on_cleanfull,
            # per-class counts for stage3
            'dire_stage3_total_real': dire_stage3_total_real,
            'dire_stage3_total_fake': dire_stage3_total_fake,
            'dire_space_full_clean_correct_real': dire_space_full_clean_correct_real,
            'dire_space_full_clean_correct_fake': dire_space_full_clean_correct_fake,
            'dire_space_stage3_adv_correct_real': dire_space_stage3_adv_correct_real,
            'dire_space_stage3_adv_correct_fake': dire_space_stage3_adv_correct_fake,
            'dire_space_stage3_adv_correct_on_cleanfull_real': dire_space_stage3_adv_correct_on_cleanfull_real,
            'dire_space_stage3_adv_correct_on_cleanfull_fake': dire_space_stage3_adv_correct_on_cleanfull_fake,
            # timing for streaming aggregate (total AA+CG per sample)
            'dire_space_total_time_sum': float(sum(dire_space_total_attack_times)) if len(dire_space_total_attack_times) > 0 else 0.0,
            'dire_space_total_time_count': int(len(dire_space_total_attack_times)),
        })

    # Add METHOD 4 results if run
    if run_zeroth:
        summary.update({
            'zeroth_clean_accuracy': zero_clean_acc,
            'zeroth_robust_accuracy': zero_robust_acc,
            'zeroth_attack_success_overall': zero_overall_success,
            'zeroth_attack_success_on_clean': zero_cond_success,
            'zeroth_clean_correct_count': zeroth_clean_correct,
            'zeroth_adv_correct_on_clean_count': zeroth_adv_correct_on_clean,
            # per-class counts
            'zeroth_total_real': zeroth_total_real,
            'zeroth_total_fake': zeroth_total_fake,
            'zeroth_clean_correct_real': zeroth_clean_correct_real,
            'zeroth_clean_correct_fake': zeroth_clean_correct_fake,
            'zeroth_adv_correct_real': zeroth_adv_correct_real,
            'zeroth_adv_correct_fake': zeroth_adv_correct_fake,
            'zeroth_adv_correct_on_clean_real': zeroth_adv_correct_on_clean_real,
            'zeroth_adv_correct_on_clean_fake': zeroth_adv_correct_on_clean_fake,
            # timing for streaming aggregate
            'zeroth_attack_time_sum': float(sum(zeroth_attack_times)) if len(zeroth_attack_times) > 0 else 0.0,
            'zeroth_attack_time_count': int(len(zeroth_attack_times)),
        })
    
    # Save summary (optionally skip in streaming); include batch suffix if provided
    if not suppress_save:
        suffix = f"_b{stream_batch_idx}" if stream_batch_idx is not None else ""
        torch.save(summary, f'{log_dir}/summary{suffix}_sd{args.seed}.pt')
    
    if not suppress_final_print:
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70 + "\n")

    if return_summary:
        return summary


def eval_stadv_dire(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    """Evaluate StAdv attack on DIRE model with linearization."""
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module
    
    x_val, y_val = x_val.to(config.device), y_val.to(config.device)
    
    print("\n" + "=" * 70)
    print("STADV EVALUATION")
    print(f"Bound: {args.adv_eps}, EOT iterations: {args.eot_iter}")
    print("=" * 70 + "\n")
    
    # Step 1: Attack classifier
    print("STEP 1: Attacking Base Classifier with StAdv")
    classifier = get_image_classifier(
        args.classifier_name,
        ckpt_path=getattr(args, 'classifier_ckpt', None)
    ).to(config.device)
    
    x_val_224 = TF.center_crop(x_val, 224)
    
    init_acc = get_accuracy(classifier, x_val_224, y_val, bs=adv_batch_size)
    print(f'Initial accuracy: {init_acc:.2%}')
    
    adversary_classifier = StAdvAttack(
        classifier, bound=args.adv_eps, num_iterations=100, eot_iter=args.eot_iter
    )
    x_adv_classifier = adversary_classifier(x_val_224, y_val)
    
    robust_acc = get_accuracy(classifier, x_adv_classifier, y_val, bs=adv_batch_size)
    print(f'Robust accuracy: {robust_acc:.2%}')
    
    torch.save([x_adv_classifier, y_val], f'{log_dir}/x_adv_classifier_sd{args.seed}.pt')
    
    # Step 2: Compute DIRE and gradient
    print("\nSTEP 2: Computing DIRE and Gradient")
    model_.reset_counter()
    model_.set_mode('full')
    model_.set_tag('compute_gradient')
    
    start_time = time.time()
    dire_output, dire_gradient = model_.compute_dire_and_gradient(x_val)
    print(f'Computation time: {time.time() - start_time:.2f}s')
    
    # Step 3: Attack linearized DIRE
    print("\nSTEP 3: Attacking Linearized DIRE Model")
    model_.reset_counter()
    model_.set_mode('linear')
    model_.set_tag('no_adv')
    
    init_acc = get_accuracy(model, x_val, y_val, bs=adv_batch_size)
    print(f'Initial accuracy (linear mode): {init_acc:.2%}')
    
    adversary_dire = StAdvAttack(
        model, bound=args.adv_eps, num_iterations=100, eot_iter=args.eot_iter
    )
    
    start_time = time.time()
    model_.reset_counter()
    model_.set_tag('attack')
    x_adv_dire = adversary_dire(x_val, y_val)
    attack_time = time.time() - start_time
    print(f'Attack completed in {attack_time:.2f}s')
    
    # Step 4: Evaluate on full DIRE
    if not getattr(args, 'quiet_full', False):
        print("\nSTEP 4: Evaluating on Full DIRE Model")
    model_.reset_counter()
    model_.set_mode('full')
    model_.set_tag('eval_adv')
    robust_acc = get_accuracy(model, x_adv_dire, y_val, bs=adv_batch_size)
    
    print(f'\n{"=" * 70}')
    print(f'FINAL RESULTS:')
    print(f'{"=" * 70}')
    print(f'Robust accuracy (full DIRE): {robust_acc:.2%}')
    print(f'{"=" * 70}\n')
    
    torch.save([x_adv_dire, y_val], f'{log_dir}/x_adv_dire_sd{args.seed}.pt')


def robustness_eval(args, config):
    """Main evaluation function."""
    middle_name = '_'.join([args.diffusion_type, args.attack_version]) \
        if args.attack_version in ['stadv', 'standard', 'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)
    
    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size
    print(f'\nSystem info:')
    print(f'  GPUs available: {ngpus}')
    print(f'  Adversarial batch size: {adv_batch_size}')
    print(f'  Device: {config.device}\n')

    # Pre-allocate GPU memory to prevent other processes from interrupting (ODE adjoint causes memory spikes)
    if torch.cuda.is_available() and getattr(args, 'reserve_memory_gb', 0) > 0:
        reserve_gb = args.reserve_memory_gb
        device_id = int(str(config.device).split(':')[-1]) if ':' in str(config.device) else 0
        reserve_bytes = int(reserve_gb * 1024**3)
        try:
            dummy = torch.empty(reserve_bytes // 4, dtype=torch.float32, device=config.device)
            torch.cuda.empty_cache()
            del dummy
            print(f'[Memory Reserve] Pre-allocated {reserve_gb}GB on GPU {device_id} to prevent external OOM\n')
        except RuntimeError as e:
            print(f'[Memory Reserve] Warning: Failed to reserve {reserve_gb}GB: {e}\n')
    
    # Load DIRE model
    print('Initializing DIRE_Adv_Model...')
    model = DIRE_Adv_Model(args, config)
    
    # Note: We don't use DataParallel for gradient computation
    # because it complicates device management during autograd.grad()
    # If you need multi-GPU, process batches sequentially instead.
    print(f'Model will run on single device: {config.device}')
    
    model = model.eval().to(config.device)
    print('Model loaded and ready.\n')
    
    # Load data (streaming or all-in-memory)
    print('Loading data...')
    if getattr(args, 'stream_data', False):
        from utils import load_data_loader
        loader, ds_size = load_data_loader(args, adv_batch_size)
        print(f'Data loader ready: {ds_size} samples')
        seen = 0
        # Global aggregators across all streamed batches
        agg = {
            'total': 0,
            'full_clean_correct': 0,
            'full_adv_correct': 0,
            'full_adv_correct_on_clean': 0,
            'full_total_real': 0,
            'full_total_fake': 0,
            'full_clean_correct_real': 0,
            'full_clean_correct_fake': 0,
            'full_adv_correct_real': 0,
            'full_adv_correct_fake': 0,
            'full_adv_correct_on_clean_real': 0,
            'full_adv_correct_on_clean_fake': 0,
            'dire_space_clean_correct': 0,
            'dire_space_stage1_adv_correct': 0,
            'dire_space_stage3_adv_correct': 0,
            'dire_space_full_clean_correct': 0,
            'dire_space_stage3_adv_correct_on_cleanfull': 0,
            'dire_stage3_total_real': 0,
            'dire_stage3_total_fake': 0,
            'dire_space_full_clean_correct_real': 0,
            'dire_space_full_clean_correct_fake': 0,
            'dire_space_stage3_adv_correct_real': 0,
            'dire_space_stage3_adv_correct_fake': 0,
            'dire_space_stage3_adv_correct_on_cleanfull_real': 0,
            'dire_space_stage3_adv_correct_on_cleanfull_fake': 0,
            'linear_clean_correct': 0,
            'linear_adv_correct': 0,
            'zeroth_clean_correct': 0,
            'zeroth_adv_correct': 0,
            'zeroth_adv_correct_on_clean': 0,
            'zeroth_total_real': 0,
            'zeroth_total_fake': 0,
            'zeroth_clean_correct_real': 0,
            'zeroth_clean_correct_fake': 0,
            'zeroth_adv_correct_real': 0,
            'zeroth_adv_correct_fake': 0,
            'zeroth_adv_correct_on_clean_real': 0,
            'zeroth_adv_correct_on_clean_fake': 0,
            # timing sums for streaming aggregate
            'full_time_sum': 0.0,
            'full_time_count': 0,
            'dire_space_total_time_sum': 0.0,
            'dire_space_total_time_count': 0,
            'zeroth_time_sum': 0.0,
            'zeroth_time_count': 0,
        }
        bidx = 0
        for xb, yb, *maybe_names in loader:
            bs_eval = 1
            if args.attack_version in ['standard', 'rand', 'custom']:
                summary = eval_autoattack_dire(
                    args, config, model, xb, yb, bs_eval, log_dir,
                    return_summary=True, suppress_save=True, suppress_final_print=True, stream_batch_idx=bidx,
                    global_start_idx=seen,  # Pass current total as start index for this batch
                    progress_prefix=agg,    # Pass cumulative counts so far for progress summary
                )
                # Aggregate by method
                total = int(summary.get('total_samples', xb.shape[0]))
                agg['total'] += total
                if args.attack_method in ['full', 'both'] and 'full_robust_accuracy' in summary:
                    # derive counts (round to int)
                    fc = int(round(summary.get('full_clean_accuracy', 0.0) * total))
                    fr = int(round(summary.get('full_robust_accuracy', 0.0) * total))
                    agg['full_clean_correct'] += fc
                    agg['full_adv_correct'] += fr
                    agg['full_adv_correct_on_clean'] += int(summary.get('full_adv_correct_on_clean_count', fr))
                    # per-class
                    agg['full_total_real'] += int(summary.get('full_total_real', 0))
                    agg['full_total_fake'] += int(summary.get('full_total_fake', 0))
                    agg['full_clean_correct_real'] += int(summary.get('full_clean_correct_real', 0))
                    agg['full_clean_correct_fake'] += int(summary.get('full_clean_correct_fake', 0))
                    agg['full_adv_correct_real'] += int(summary.get('full_adv_correct_real', 0))
                    agg['full_adv_correct_fake'] += int(summary.get('full_adv_correct_fake', 0))
                    agg['full_adv_correct_on_clean_real'] += int(summary.get('full_adv_correct_on_clean_real', 0))
                    agg['full_adv_correct_on_clean_fake'] += int(summary.get('full_adv_correct_on_clean_fake', 0))
                    # timing aggregate if provided
                    agg['full_time_sum'] += float(summary.get('full_attack_time_sum', 0.0) or 0.0)
                    agg['full_time_count'] += int(summary.get('full_attack_time_count', 0) or 0)
                if args.attack_method == 'dire_space' and 'dire_space_stage3_robust_accuracy' in summary:
                    agg['dire_space_clean_correct'] += int(round(summary.get('dire_space_clean_accuracy', 0.0) * total))
                    agg['dire_space_stage1_adv_correct'] += int(round(summary.get('dire_space_stage1_robust_accuracy', 0.0) * total))
                    agg['dire_space_stage3_adv_correct'] += int(round(summary.get('dire_space_stage3_robust_accuracy', 0.0) * total))
                    agg['dire_space_full_clean_correct'] += int(summary.get('dire_space_full_clean_correct_count', 0))
                    agg['dire_space_stage3_adv_correct_on_cleanfull'] += int(summary.get('dire_space_stage3_adv_correct_on_cleanfull_count', 0))
                    # per-class
                    agg['dire_stage3_total_real'] += int(summary.get('dire_stage3_total_real', 0))
                    agg['dire_stage3_total_fake'] += int(summary.get('dire_stage3_total_fake', 0))
                    agg['dire_space_full_clean_correct_real'] += int(summary.get('dire_space_full_clean_correct_real', 0))
                    agg['dire_space_full_clean_correct_fake'] += int(summary.get('dire_space_full_clean_correct_fake', 0))
                    agg['dire_space_stage3_adv_correct_real'] += int(summary.get('dire_space_stage3_adv_correct_real', 0))
                    agg['dire_space_stage3_adv_correct_fake'] += int(summary.get('dire_space_stage3_adv_correct_fake', 0))
                    agg['dire_space_stage3_adv_correct_on_cleanfull_real'] += int(summary.get('dire_space_stage3_adv_correct_on_cleanfull_real', 0))
                    agg['dire_space_stage3_adv_correct_on_cleanfull_fake'] += int(summary.get('dire_space_stage3_adv_correct_on_cleanfull_fake', 0))
                    # timing aggregate if provided
                    agg['dire_space_total_time_sum'] += float(summary.get('dire_space_total_time_sum', 0.0) or 0.0)
                    agg['dire_space_total_time_count'] += int(summary.get('dire_space_total_time_count', 0) or 0)
                if args.attack_method in ['linear', 'both'] and 'linear_robust_accuracy' in summary:
                    agg['linear_clean_correct'] += int(round(summary.get('linear_clean_accuracy', 0.0) * total))
                    agg['linear_adv_correct'] += int(round(summary.get('linear_robust_accuracy', 0.0) * total))
                if args.attack_method == 'zeroth' and 'zeroth_robust_accuracy' in summary:
                    agg['zeroth_clean_correct'] += int(round(summary.get('zeroth_clean_accuracy', 0.0) * total))
                    agg['zeroth_adv_correct'] += int(round(summary.get('zeroth_robust_accuracy', 0.0) * total))
                    agg['zeroth_adv_correct_on_clean'] += int(summary.get('zeroth_adv_correct_on_clean_count', 0))
                    # per-class
                    agg['zeroth_total_real'] += int(summary.get('zeroth_total_real', 0))
                    agg['zeroth_total_fake'] += int(summary.get('zeroth_total_fake', 0))
                    agg['zeroth_clean_correct_real'] += int(summary.get('zeroth_clean_correct_real', 0))
                    agg['zeroth_clean_correct_fake'] += int(summary.get('zeroth_clean_correct_fake', 0))
                    agg['zeroth_adv_correct_real'] += int(summary.get('zeroth_adv_correct_real', 0))
                    agg['zeroth_adv_correct_fake'] += int(summary.get('zeroth_adv_correct_fake', 0))
                    agg['zeroth_adv_correct_on_clean_real'] += int(summary.get('zeroth_adv_correct_on_clean_real', 0))
                    agg['zeroth_adv_correct_on_clean_fake'] += int(summary.get('zeroth_adv_correct_on_clean_fake', 0))
                    # timing aggregate if provided
                    agg['zeroth_time_sum'] += float(summary.get('zeroth_attack_time_sum', 0.0) or 0.0)
                    agg['zeroth_time_count'] += int(summary.get('zeroth_attack_time_count', 0) or 0)
            elif args.attack_version == 'stadv':
                # For now, stream stadv without aggregation changes
                eval_stadv_dire(args, config, model, xb, yb, bs_eval, log_dir)
            else:
                raise NotImplementedError(f'Unknown attack_version: {args.attack_version}')
            seen += xb.shape[0]
            bidx += 1
            del xb, yb
            torch.cuda.empty_cache()
        print(f'[stream] processed {seen}/{ds_size} samples')

        # Print aggregate results once
        print("\n" + "=" * 70)
        print("FINAL RESULTS (STREAM AGGREGATE)")
        print("=" * 70)
        print(f"Total samples evaluated: {agg['total']}")
        # Prepare an aggregate summary dict to optionally save
        agg_summary = {
            'attack_version': args.attack_version,
            'attack_method': args.attack_method,
            'lp_norm': args.lp_norm,
            'epsilon': args.adv_eps,
            'total_samples': agg['total'],
        }
        if args.attack_method in ['full', 'both'] and agg['total'] > 0 and not getattr(args, 'quiet_full', False):
            full_clean_acc = agg['full_clean_correct'] / agg['total'] if agg['total'] else 0.0
            full_robust_acc = agg['full_adv_correct'] / agg['total'] if agg['total'] else 0.0
            full_overall_success = 1 - full_robust_acc
            full_cond_success = (1 - (agg['full_adv_correct_on_clean'] / agg['full_clean_correct'])) if agg['full_clean_correct'] > 0 else 0.0
            print("METHOD 2: Full DIRE Attack")
            print(f"  Clean accuracy:   {full_clean_acc:.2%} ({agg['full_clean_correct']}/{agg['total']})")
            print(f"  Robust accuracy:  {full_robust_acc:.2%} ({agg['full_adv_correct']}/{agg['total']})")
            print(f"  Attack success (overall):        {full_overall_success:.2%}")
            print(f"  Attack success (on clean-only):  {full_cond_success:.2%}")
            # Average attack time
            if agg.get('full_time_count', 0) > 0:
                avg_attack_time = agg['full_time_sum'] / max(agg['full_time_count'], 1)
                print(f"  Average attack time per sample:  {avg_attack_time:.2f}s")
            # Per-class aggregate
            if (agg['full_total_real'] + agg['full_total_fake']) > 0:
                if agg['full_total_real'] > 0:
                    real_overall_succ = 1 - (agg['full_adv_correct_real'] / max(agg['full_total_real'], 1))
                    real_clean_cond_succ = (1 - (agg['full_adv_correct_on_clean_real'] / max(agg['full_clean_correct_real'], 1))) if agg['full_clean_correct_real'] > 0 else 0.0
                    print(f"  [by-class] real Success(overall): {real_overall_succ:.2%} | Success(on clean-only): {real_clean_cond_succ:.2%}")
                if agg['full_total_fake'] > 0:
                    fake_overall_succ = 1 - (agg['full_adv_correct_fake'] / max(agg['full_total_fake'], 1))
                    fake_clean_cond_succ = (1 - (agg['full_adv_correct_on_clean_fake'] / max(agg['full_clean_correct_fake'], 1))) if agg['full_clean_correct_fake'] > 0 else 0.0
                    print(f"  [by-class] fake Success(overall): {fake_overall_succ:.2%} | Success(on clean-only): {fake_clean_cond_succ:.2%}")
            agg_summary.update({
                'full_clean_accuracy': full_clean_acc,
                'full_robust_accuracy': full_robust_acc,
                'full_attack_success_overall': full_overall_success,
                'full_attack_success_on_clean': full_cond_success,
                'full_clean_correct_count': agg['full_clean_correct'],
                'full_adv_correct_on_clean_count': agg['full_adv_correct_on_clean'],
            })
        if args.attack_method == 'dire_space' and agg['total'] > 0:
            ds_clean_acc = agg['dire_space_clean_correct'] / agg['total']
            stage1_acc = agg['dire_space_stage1_adv_correct'] / agg['total']
            stage3_acc = agg['dire_space_stage3_adv_correct'] / agg['total']
            stage3_overall_success = 1 - stage3_acc
            stage3_cond_success = (1 - (agg['dire_space_stage3_adv_correct_on_cleanfull'] / agg['dire_space_full_clean_correct'])) if agg['dire_space_full_clean_correct'] > 0 else 0.0
            print("METHOD 3: DIRE-Space Attack")
            print(f"  Clean accuracy:           {ds_clean_acc:.2%} ({agg['dire_space_clean_correct']}/{agg['total']})")
            print(f"  Stage1 robust (DIRE map): {stage1_acc:.2%} ({agg['dire_space_stage1_adv_correct']}/{agg['total']})")
            print(f"  Stage3 robust (full):     {stage3_acc:.2%} ({agg['dire_space_stage3_adv_correct']}/{agg['total']})")
            print(f"  Stage3 success (overall): {stage3_overall_success:.2%}")
            print(f"  Stage3 success (on clean-only, full-pipeline clean): {stage3_cond_success:.2%}")
            # Average total attack time per sample
            if agg.get('dire_space_total_time_count', 0) > 0:
                avg_attack_time = agg['dire_space_total_time_sum'] / max(agg['dire_space_total_time_count'], 1)
                print(f"  Average total attack time per sample:  {avg_attack_time:.2f}s")
            # Per-class aggregate for Stage3
            if (agg['dire_stage3_total_real'] + agg['dire_stage3_total_fake']) > 0:
                if agg['dire_stage3_total_real'] > 0:
                    real_stage3_overall_succ = 1 - (agg['dire_space_stage3_adv_correct_real'] / max(agg['dire_stage3_total_real'], 1))
                    real_stage3_cond_succ = (1 - (agg['dire_space_stage3_adv_correct_on_cleanfull_real'] / max(agg['dire_space_full_clean_correct_real'], 1))) if agg['dire_space_full_clean_correct_real'] > 0 else 0.0
                    print(f"  [by-class] real Stage3 Success(overall): {real_stage3_overall_succ:.2%} | Stage3 Success(on clean-only): {real_stage3_cond_succ:.2%}")
                if agg['dire_stage3_total_fake'] > 0:
                    fake_stage3_overall_succ = 1 - (agg['dire_space_stage3_adv_correct_fake'] / max(agg['dire_stage3_total_fake'], 1))
                    fake_stage3_cond_succ = (1 - (agg['dire_space_stage3_adv_correct_on_cleanfull_fake'] / max(agg['dire_space_full_clean_correct_fake'], 1))) if agg['dire_space_full_clean_correct_fake'] > 0 else 0.0
                    print(f"  [by-class] fake Stage3 Success(overall): {fake_stage3_overall_succ:.2%} | Stage3 Success(on clean-only): {fake_stage3_cond_succ:.2%}")
            agg_summary.update({
                'dire_space_clean_accuracy': ds_clean_acc,
                'dire_space_stage1_robust_accuracy': stage1_acc,
                'dire_space_stage3_robust_accuracy': stage3_acc,
                'dire_space_stage3_success_overall': stage3_overall_success,
                'dire_space_stage3_success_on_clean_full': stage3_cond_success,
                'dire_space_full_clean_correct_count': agg['dire_space_full_clean_correct'],
                'dire_space_stage3_adv_correct_on_cleanfull_count': agg['dire_space_stage3_adv_correct_on_cleanfull'],
            })
        if args.attack_method == 'zeroth' and agg['total'] > 0:
            zero_clean_acc = agg['zeroth_clean_correct'] / agg['total']
            zero_robust_acc = agg['zeroth_adv_correct'] / agg['total']
            zero_overall_success = 1 - zero_robust_acc
            zero_cond_success = (1 - (agg['zeroth_adv_correct_on_clean'] / agg['zeroth_clean_correct'])) if agg['zeroth_clean_correct'] > 0 else 0.0
            print("METHOD 4: Zeroth-Order Black-Box Attack")
            print(f"  Clean accuracy:   {zero_clean_acc:.2%} ({agg['zeroth_clean_correct']}/{agg['total']})")
            print(f"  Robust accuracy:  {zero_robust_acc:.2%} ({agg['zeroth_adv_correct']}/{agg['total']})")
            print(f"  Attack success (overall):        {zero_overall_success:.2%}")
            print(f"  Attack success (on clean-only):  {zero_cond_success:.2%}")
            # Average attack time per sample
            if agg.get('zeroth_time_count', 0) > 0:
                avg_attack_time = agg['zeroth_time_sum'] / max(agg['zeroth_time_count'], 1)
                print(f"  Average attack time per sample:  {avg_attack_time:.2f}s")
            # Per-class aggregate
            if (agg['zeroth_total_real'] + agg['zeroth_total_fake']) > 0:
                if agg['zeroth_total_real'] > 0:
                    real_overall_succ = 1 - (agg['zeroth_adv_correct_real'] / max(agg['zeroth_total_real'], 1))
                    real_clean_cond_succ = (1 - (agg['zeroth_adv_correct_on_clean_real'] / max(agg['zeroth_clean_correct_real'], 1))) if agg['zeroth_clean_correct_real'] > 0 else 0.0
                    print(f"  [by-class] real Success(overall): {real_overall_succ:.2%} | Success(on clean-only): {real_clean_cond_succ:.2%}")
                if agg['zeroth_total_fake'] > 0:
                    fake_overall_succ = 1 - (agg['zeroth_adv_correct_fake'] / max(agg['zeroth_total_fake'], 1))
                    fake_clean_cond_succ = (1 - (agg['zeroth_adv_correct_on_clean_fake'] / max(agg['zeroth_clean_correct_fake'], 1))) if agg['zeroth_clean_correct_fake'] > 0 else 0.0
                    print(f"  [by-class] fake Success(overall): {fake_overall_succ:.2%} | Success(on clean-only): {fake_clean_cond_succ:.2%}")
            agg_summary.update({
                'zeroth_clean_accuracy': zero_clean_acc,
                'zeroth_robust_accuracy': zero_robust_acc,
                'zeroth_attack_success_overall': zero_overall_success,
                'zeroth_attack_success_on_clean': zero_cond_success,
                'zeroth_clean_correct_count': agg['zeroth_clean_correct'],
                'zeroth_adv_correct_on_clean_count': agg['zeroth_adv_correct_on_clean'],
            })
        print("=" * 70 + "\n")
        # Save aggregate summary for downstream analysis
        # Enrich and save aggregate summary for downstream analysis
        try:
            # Per-class aggregates if available
            agg_summary.update({
                'full_total_real': agg.get('full_total_real', 0),
                'full_total_fake': agg.get('full_total_fake', 0),
                'full_clean_correct_real': agg.get('full_clean_correct_real', 0),
                'full_clean_correct_fake': agg.get('full_clean_correct_fake', 0),
                'full_adv_correct_real': agg.get('full_adv_correct_real', 0),
                'full_adv_correct_fake': agg.get('full_adv_correct_fake', 0),
                'full_adv_correct_on_clean_real': agg.get('full_adv_correct_on_clean_real', 0),
                'full_adv_correct_on_clean_fake': agg.get('full_adv_correct_on_clean_fake', 0),
                'dire_stage3_total_real': agg.get('dire_stage3_total_real', 0),
                'dire_stage3_total_fake': agg.get('dire_stage3_total_fake', 0),
                'dire_space_full_clean_correct_real': agg.get('dire_space_full_clean_correct_real', 0),
                'dire_space_full_clean_correct_fake': agg.get('dire_space_full_clean_correct_fake', 0),
                'dire_space_stage3_adv_correct_real': agg.get('dire_space_stage3_adv_correct_real', 0),
                'dire_space_stage3_adv_correct_fake': agg.get('dire_space_stage3_adv_correct_fake', 0),
                'dire_space_stage3_adv_correct_on_cleanfull_real': agg.get('dire_space_stage3_adv_correct_on_cleanfull_real', 0),
                'dire_space_stage3_adv_correct_on_cleanfull_fake': agg.get('dire_space_stage3_adv_correct_on_cleanfull_fake', 0),
                'zeroth_total_real': agg.get('zeroth_total_real', 0),
                'zeroth_total_fake': agg.get('zeroth_total_fake', 0),
                'zeroth_clean_correct_real': agg.get('zeroth_clean_correct_real', 0),
                'zeroth_clean_correct_fake': agg.get('zeroth_clean_correct_fake', 0),
                'zeroth_adv_correct_real': agg.get('zeroth_adv_correct_real', 0),
                'zeroth_adv_correct_fake': agg.get('zeroth_adv_correct_fake', 0),
                'zeroth_adv_correct_on_clean_real': agg.get('zeroth_adv_correct_on_clean_real', 0),
                'zeroth_adv_correct_on_clean_fake': agg.get('zeroth_adv_correct_on_clean_fake', 0),
            })
            torch.save(agg_summary, f"{log_dir}/summary_aggregate_sd{args.seed}.pt")
            print(f"[stream] aggregate summary saved to: {log_dir}/summary_aggregate_sd{args.seed}.pt")
        except Exception as e:
            print(f"[stream] warning: failed to save aggregate summary: {e}")
        logger.close()
        return
    else:
        from utils import load_data
        x_val, y_val = load_data(args, adv_batch_size)
        print(f'Data loaded: {x_val.shape}, labels: {y_val.shape}')
        print(f'Data range: [{x_val.min().item():.4f}, {x_val.max().item():.4f}]\n')
        # Verify data is in correct format
        expected_size = int(getattr(args, 'dire_image_size', 256))
        assert x_val.shape[-2:] == (expected_size, expected_size), \
            f"Expected {expected_size}×{expected_size} images, got {x_val.shape[-2:]}. " \
            f"Make sure preprocessing applies Resize({expected_size}) + CenterCrop({expected_size})!"
        assert x_val.min() >= 0 and x_val.max() <= 1, \
            f"Expected [0, 1] range, got [{x_val.min():.4f}, {x_val.max():.4f}]"
        # Non-streaming run
        if args.attack_version in ['standard', 'rand', 'custom']:
            eval_autoattack_dire(args, config, model, x_val, y_val, adv_batch_size, log_dir,
                               suppress_save=(not args.save_pt))
        elif args.attack_version == 'stadv':
            eval_stadv_dire(args, config, model, x_val, y_val, adv_batch_size, log_dir)
        else:
            raise NotImplementedError(f'Unknown attack_version: {args.attack_version}')
    
    logger.close()


def parse_args_and_config():
    """Parse command line arguments and configuration file."""
    parser = argparse.ArgumentParser(
        description='DIRE Adversarial Robustness Evaluation with Linearization'
    )
    
    # Model configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the config file (e.g., imagenet.yml)')
    parser.add_argument('--diffusion_model_path', type=str, required=True,
                       help='Path to diffusion model checkpoint')
    
    # DIRE ODE parameters (matching export_dire.py defaults)
    parser.add_argument('--diffusion_type', type=str, default='ode',
                       help='Diffusion type (use "ode" for DIRE)')
    parser.add_argument('--dire_image_size', type=int, default=256,
                       help='Resolution expected by the diffusion model (e.g., 256 or 64). Controls input resize/crop and input-size assertions. Classifier preprocessing will upsample DIRE output to 256 before CenterCrop(224) to match training.')
    parser.add_argument('--t', type=int, default=50,
                       help='Number of ODE time steps')
    parser.add_argument('--ode_method', type=str, default='euler',
                       help='ODE solver method')
    parser.add_argument('--ode_step_size', type=float, default=2e-2,
                       help='ODE step size')
    parser.add_argument('--ode_rtol', type=float, default=1e-3,
                       help='ODE relative tolerance')
    parser.add_argument('--ode_atol', type=float, default=1e-3,
                       help='ODE absolute tolerance')
    parser.add_argument('--fix_rand', type=str2bool, default=True,
                       help='Use fixed randomness for reproducibility')
    parser.add_argument('--eot_reps', type=int, default=1,
                       help='EOT repetitions for DIRE')
    parser.add_argument('--ode_dire_smooth_eps', type=float, default=5e-5,
                       help='DIRE smoothing epsilon')
    
    # Classifier preprocessing (must match training config!)
    parser.add_argument('--dire_apply_imagenet_norm', type=str2bool, default=True,
                       help='Apply ImageNet normalization to DIRE maps (default in training)')
    parser.add_argument('--dire_scale_to_m11', type=str2bool, default=False,
                       help='Scale DIRE maps to [-1,1] instead of ImageNet norm')
    
    # Dataset and classifier
    parser.add_argument('--domain', type=str, default='imagenet',
                       help='Dataset domain')
    parser.add_argument('--classifier_name', type=str, required=True,
                       help='Classifier to defend')
    parser.add_argument('--classifier_ckpt', type=str, default=None,
                       help='Path to classifier checkpoint')
    parser.add_argument('--partition', type=str, default='val',
                       help='Dataset partition')
    parser.add_argument('--num_sub', type=int, default=1000,
                       help='Number of samples to evaluate')
    
    # Data loading (custom path support)
    parser.add_argument('--data_root', type=str, default=None,
                       help='Custom data root directory')
    parser.add_argument('--real_subdir', type=str, default='real/real',
                       help='Real images subdirectory')
    parser.add_argument('--fake_dirs', type=str, default=None,
                       help='Comma-separated fake image subdirectories')
    parser.add_argument('--glob_exts', type=str, default='.png,.jpg,.jpeg,.bmp,.webp',
                       help='Image extensions to include')
    parser.add_argument('--real_offset', type=int, default=0,
                       help='Offset for real images')
    parser.add_argument('--real_count', type=int, default=None,
                       help='Number of real images to use')
    parser.add_argument('--fake_offset', type=int, default=0,
                       help='Offset for fake images')
    parser.add_argument('--fake_count', type=int, default=None,
                       help='Number of fake images to use')
    parser.add_argument('--real_label', type=int, default=0,
                       help='Label for real images')
    parser.add_argument('--fake_label', type=int, default=1,
                       help='Label for fake images')
    parser.add_argument('--only_real', action='store_true',
                       help='Use only real images')
    parser.add_argument('--only_fake', action='store_true',
                       help='Use only fake images')
    parser.add_argument('--pick_real', type=int, default=None,
                       help='Pick specific number of real images')
    parser.add_argument('--pick_fake', type=int, default=None,
                       help='Pick specific number of fake images')
    parser.add_argument('--stream_data', type=str2bool, default=False,
                       help='Stream data by DataLoader to avoid loading all samples into memory at once')
    parser.add_argument('--dataloader_workers', type=int, default=4,
                       help='Number of DataLoader workers when streaming data')
    
    # Attack configuration
    parser.add_argument('--attack_version', type=str, default='custom',
                       choices=['standard', 'rand', 'custom', 'stadv'],
                       help='Attack version to use')
    parser.add_argument('--attack_type', type=str, default='apgd-ce',
                       help='Attack types (comma-separated for custom version)')
    parser.add_argument('--attack_method', type=str, default='full',
                       choices=['full', 'dire_space', 'zeroth'],
                       help='Attack method: full (direct attack on complete DIRE pipeline), dire_space (attack classifier in DIRE output space, then CG backproject to input), zeroth (black-box zeroth-order optimization directly on input). Note: "linear" and "both" modes have been deprecated due to inaccurate diagonal approximation.')
    parser.add_argument('--quiet_full', type=str2bool, default=True,
                       help='Silence per-sample and progress prints for FULL DIRE attack; keep final summary.')
    parser.add_argument('--lp_norm', type=str, default='Linf',
                       choices=['Linf', 'L2'],
                       help='Lp norm for attack')
    parser.add_argument('--adv_eps', type=float, default=0.03,
                       help='Attack epsilon')
    parser.add_argument('--adv_batch_size', type=int, default=1,
                       help='Batch size for attack (use 1 to avoid OOM during gradient computation)')
    parser.add_argument('--eot_iter', type=int, default=20,
                       help='EOT iterations for rand attack')
    parser.add_argument('--autoattack_n_iter', type=int, default=100,
                       help='Number of iterations for AutoAttack APGD')
    
    # CG backprojection parameters
    parser.add_argument('--backproj_cg_iters', type=int, default=100,
                       help='Maximum CG iterations for backprojection in dire_space mode')
    parser.add_argument('--backproj_cg_lambda', type=float, default=1e-4,
                       help='Tikhonov regularization for CG backprojection')
    parser.add_argument('--backproj_cg_tol', type=float, default=1e-6,
                       help='Convergence tolerance for CG early stopping (on residual norm)')
    parser.add_argument('--backproj_smooth_sigma', type=float, default=0.0,
                       help='Optional Gaussian smoothing sigma applied to backprojected delta before eps clip (0 disables)')
    parser.add_argument('--backproj_smooth_ksize', type=int, default=3,
                       help='Kernel size for Gaussian smoothing (odd integer, >=3)')
    parser.add_argument('--backproj_cache_mode', type=str, default='none', choices=['none', 'matrix'],
                       help='Backprojection acceleration: none (CG only) or matrix (precompute dense JTJ and solve).')
    parser.add_argument('--backproj_solver', type=str, default='cg', choices=['cg','lsqr','gmres'],
                        help='Linear solver for backprojection when cache_mode=none: cg (on normal equations), lsqr (Golub-Kahan), or gmres (restarted GMRES on J).')
    parser.add_argument('--backproj_gmres_restart', type=int, default=50,
                        help='Restart parameter m for GMRES (basis size).')
    parser.add_argument('--backproj_matrix_device', type=str, default='cpu',
                       help='Device to hold JTJ and solve in matrix mode (cpu or cuda:<id>).')
    parser.add_argument('--backproj_matrix_max_n', type=int, default=4096,
                       help='Safety cap on n=C*H*W for dense JTJ build. For 32x32x3, n=3072.')
    parser.add_argument('--backproj_matrix_vectorize', type=str2bool, default=False,
                        help='Try vectorized Jacobian computation (may be unsupported with some custom autograd ops).')
    parser.add_argument('--backproj_matrix_dtype', type=str, default='fp32', choices=['fp32','fp16','bf16'],
                        help='Dtype for dense Jacobian in matrix mode (fp16/bf16 reduce memory, may affect accuracy).')
    parser.add_argument('--save_adv_images', type=str2bool, default=True,
                       help='Save per-sample adversarial images (PNG) in log_dir/adv_images')
    parser.add_argument('--save_pt', type=str2bool, default=True,
                       help='Save adversarial tensors as .pt files (can be disabled to save storage)')
    parser.add_argument('--save_grad_probe_dir', type=str, default='',
                       help='If set, save per-sample full-mode grad-probe input gradients to this directory')
    parser.add_argument('--grad_probe_only', type=str2bool, default=False,
                       help='Only run clean prediction and grad-probe, then skip attack generation')

    # Zeroth-order (black-box) attack parameters
    parser.add_argument('--zo_iters', type=int, default=100,
                        help='Zeroth-order attack iterations')
    parser.add_argument('--zo_q', type=int, default=64,
                        help='Number of random directions per iteration for zeroth-order gradient estimation')
    parser.add_argument('--zo_batch_q', type=int, default=64,
                        help='Parallel query batch size per iteration for zeroth-order estimation (<= zo_q)')
    parser.add_argument('--zo_objective', type=str, default='margin', choices=['ce', 'margin', 'dlr'],
                        help='Objective used for zeroth-order attack: ce (cross-entropy), margin (max other - true), dlr (decoupled logit ratio)')
    parser.add_argument('--zo_mu', type=float, default=0.01,
                        help='Finite-difference smoothing parameter (perturbation radius) for zeroth-order gradient estimate')
    parser.add_argument('--zo_alpha', type=float, default=0.01,
                        help='Step size for zeroth-order attack updates')
    parser.add_argument('--zo_symmetric', type=str2bool, default=True,
                        help='Use symmetric finite differences ((f(x+mu)-f(x-mu))/(2*mu)); when False, use one-sided (f(x+mu)-f(x))/mu')
    parser.add_argument('--zo_interior_bias', type=float, default=0.0,
                        help='Shift query base slightly inward from [0,1] bounds for zeroth queries (e.g., 1e-3) to reduce clamp-induced cancellations')
    parser.add_argument('--zo_query_clip_eps', type=float, default=0.0,
                        help='Soft query clip margin: clamp queries to [eps, 1-eps] to avoid identical clamped samples')
    parser.add_argument('--zo_early_stop_on_success', type=str2bool, default=True,
                        help='Early stop zeroth-order iterations once misclassification is achieved')
    parser.add_argument('--zo_early_patience', type=int, default=100,
                        help='Patience (iterations) for loss-plateau early stopping')
    parser.add_argument('--zo_early_min_delta', type=float, default=1e-6,
                        help='Minimum loss increase to reset patience (we maximize loss)')
    parser.add_argument('--zo_adapt_mu', type=str2bool, default=True,
                        help='Adaptively increase mu if finite-difference signal is too small')
    parser.add_argument('--zo_adapt_mu_factor', type=float, default=2.0,
                        help='Multiplicative factor to increase mu when adapting')
    parser.add_argument('--zo_adapt_mu_min_diff', type=float, default=1e-4,
                        help='If mean |f(x+mu)-f(x-mu)| < this threshold, increase mu')
    parser.add_argument('--zo_adapt_mu_max', type=float, default=0.05,
                        help='Upper bound for adaptive mu to avoid overshooting')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0,
                       help='Data loading seed')
    parser.add_argument('--exp', type=str, default='exp',
                       help='Experiment directory')
    parser.add_argument('-i', '--image_folder', type=str, default='images',
                       help='Output folder name')
    parser.add_argument('--verbose', type=str, default='info',
                       help='Logging verbosity')
    parser.add_argument('--ni', action='store_true',
                       help='No interaction mode')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (e.g., cuda:0, cuda:3, cpu). Auto-detect if not specified.')
    parser.add_argument('--reserve_memory_gb', type=float, default=0,
                       help='Pre-allocate GPU memory (GB) to prevent OOM from external processes during ODE adjoint. Set to 0 to disable.')
    parser.add_argument('--deterministic', type=str2bool, default=True,
                       help='Enable deterministic modes (slower but reproducible). Set False for faster, non-deterministic cuDNN.')

    args = parser.parse_args()
    
    # Load config file
    config_path = os.path.join('configs', args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)
    
    # Setup logging
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f'Invalid log level: {args.verbose}')
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
    
    # Setup output directory
    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)
    
    # Setup device (honor --device if provided, otherwise auto-detect)
    if args.device:
        device = torch.device(args.device)
        logging.info(f"Using specified device: {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Auto-detected device: {device}")
    config.device = device
    
    # Determinism vs speed toggle
    parser_deterministic = True  # default legacy behavior
    try:
        parser_deterministic = bool(getattr(args, 'deterministic', True))
    except Exception:
        parser_deterministic = True
    if parser_deterministic:
        print("\n" + "=" * 70)
        print("Setting deterministic environment (matching export_dire.py)")
        print("=" * 70)
        make_deterministic(args.seed)
        print("Deterministic mode enabled.")
        print("=" * 70 + "\n")
    else:
        # Prefer speed: allow non-deterministic algorithms and enable cudnn benchmark
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        print("[perf] Non-deterministic fast mode enabled (cuDNN benchmark on)")
    
    return args, config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    robustness_eval(args, config)
