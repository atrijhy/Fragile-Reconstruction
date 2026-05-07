# ---------------------------------------------------------------
# DIRE Attack Model wrapper: logits = classifier(|x - recon(x)|)
# 依赖 GuidedDiffusionDDIM 作为可微分的reconstruction后端
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from runners.diffpure_guided_ddim import GuidedDiffusionDDIM
from runners.dire_ode_model import DIRE_ODE_Model, ODEDireConfig
from utils import get_image_classifier


class DIRE_Adv_Model(nn.Module):
    """
    将DIRE特征接入分类器：
      forward(x):
        x_in in [0,1] -> map到[-1,1] -> dire_map -> map回[0,1] (如有需要) -> classifier
    注意：为了端到端可微，内部使用enable_grad=True。

    后端选择：
    - DDIM（默认）：保持原有可微 DDIM 路径。
    - ODE：使用概率流 ODE（伴随法）作为重建后端，更省显存。
      通过 args.dire_backend in {"ddim", "ode"} 选择，默认 "ddim"。
    """
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = getattr(config, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 分类器
        self.classifier = get_image_classifier(
            args.classifier_name,
            ckpt_path=getattr(args, 'classifier_ckpt', None)
        ).to(self.device)
        # Ensure eval mode for inference/eval; prevents BN/Dropout updates.
        try:
            self.classifier.eval()
        except Exception:
            pass

        self.backend = getattr(args, 'dire_backend', 'ddim')
        if self.backend == 'ddim':
            self.recon_model = GuidedDiffusionDDIM(args, config, device=self.device)
        elif self.backend == 'ode':
            ode_cfg = ODEDireConfig(
                t_steps=getattr(args, 't', 50),
                method=getattr(args, 'ode_method', 'euler'),
                rtol=getattr(args, 'ode_rtol', 1e-3),
                atol=getattr(args, 'ode_atol', 1e-3),
                step_size=getattr(args, 'ode_step_size', 2e-2),
                fix_rand=getattr(args, 'fix_rand', True),
                eot_reps=getattr(args, 'eot_reps', 1),
                clamp_recon=True,
                dire_scale_to_01=True,
                dire_smooth_eps=float(getattr(args, 'ode_dire_smooth_eps', 5e-5)),
                dire_log_grad=bool(getattr(args, 'ode_dire_log_grad', False)),
                force_fp32=bool(getattr(args, 'ode_force_fp32', False)),
                nan_fallback=bool(getattr(args, 'ode_nan_fallback', True)),
                use_unet_checkpoint=bool(getattr(args, 'ode_use_unet_checkpoint', True)),
                disable_solver_fallback=bool(getattr(args, 'ode_disable_solver_fallback', True)),
                local_half=bool(getattr(args, 'ode_local_half', True)),
                # Freezing params breaks guided_diffusion's custom backward used by adjoint VJP; keep False by default.
                freeze_model_params=bool(getattr(args, 'ode_freeze_params', False)),
                empty_cache_each_iter=bool(getattr(args, 'ode_empty_cache_each_iter', True)),
            )
            self.recon_model = DIRE_ODE_Model(args, config, ode_cfg, device=self.device)
            # One-time debug print to ensure runtime alignment of ODE settings
            try:
                print("[DIRE_Adv][ode] cfg:", {
                    't_steps': ode_cfg.t_steps,
                    'method': ode_cfg.method,
                    'step_size': ode_cfg.step_size,
                    'rtol': ode_cfg.rtol,
                    'atol': ode_cfg.atol,
                    'fix_rand': ode_cfg.fix_rand,
                    'eot_reps': ode_cfg.eot_reps,
                    'dire_smooth_eps': getattr(ode_cfg, 'dire_smooth_eps', None),
                })
            except Exception:
                pass
        else:
            raise NotImplementedError(f"unknown dire_backend: {self.backend}")

        # 计数器与可选 tag（与 SDE_Adv_Model 风格一致）
        self.register_buffer('counter', torch.zeros(1, device=self.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        """x ∈ [0,1] -> DIRE logits.
        
        CRITICAL: DIRE 计算应该在 256×256 上进行（与官方 compute_dire.py 对齐）
        - 输入 x 应该已经是 256×256（从 load_data 获得，不经过 CenterCrop(224)）
        - 计算 DIRE map 后，再下采样到 224 送入分类器
        - 避免 224→256→224 的双重插值误差
        """
        counter = self.counter.item()
        # 日志频率控制：在开始时（0）打印一次，其后每 log_prob_every 次打印
        log_every = getattr(self.args, 'log_prob_every', 10)
        should_log = (counter == 0) or (log_every > 0 and (counter % log_every == 0))
        if should_log:
            print(f'diffusion times: {counter}')

        # CRITICAL 修复：输入应该已经是 256×256，不需要上采样
        # 如果从 load_data 获得的是 224×224，说明 load_data 的 transform 需要修复
        is_imagenet = ('imagenet' in getattr(self.args, 'domain', ''))
        x_in = x
        if is_imagenet and x.shape[-1] != 256:
            # 警告：这不应该发生！load_data 应该输出 256×256
            print(f"[warn] Input size {x.shape[-2:]} != 256, upsampling (this adds interpolation error!)")
            x_in = F.interpolate(x_in, size=(256, 256), mode='bilinear', align_corners=False)

        # x: [0,1]，转换到[-1,1]
        x_m1_1 = (x_in - 0.5) * 2.0

        # 计算 DIRE map（可微分），统一调用
        import time
        start_time = time.time()
        dire = self.recon_model.dire_map(x_m1_1, enable_grad=True, normalize=True)
        dire_01 = torch.clamp(dire, 0.0, 1.0)
        minutes, seconds = divmod(time.time() - start_time, 60)

        # 回到分类器输入分辨率：256 -> 224
        # 使用 torchvision.transforms.functional.center_crop（与官方 compute_dire.py 一致）
        if is_imagenet:
            dire_01 = TF.center_crop(dire_01, 224)

        if should_log:
            print(f'x shape (before recon): {x.shape}')
            print(f'dire_map shape (before classifier): {dire_01.shape}')
            try:
                _min = float(dire_01.min().item())
                _max = float(dire_01.max().item())
                _mean = float(dire_01.mean().item())
                print(f'dire_map stats: min={_min:.6f} max={_max:.6f} mean={_mean:.6f}')
            except Exception:
                pass
            print("DIRE compute time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        logits = self.classifier(dire_01)

        def _prob_tensor_to_binary_logits(prob_tensor: torch.Tensor) -> torch.Tensor:
            probs = prob_tensor
            if torch.any((probs < 0) | (probs > 1)):
                probs = torch.sigmoid(probs)
            eps = 1e-6
            probs = probs * (1.0 - 2.0 * eps) + eps
            logits_real = torch.log1p(-probs)
            logits_fake = torch.log(probs)
            return torch.stack([logits_real, logits_fake], dim=1)

        fake_prob_index = getattr(self.args, 'fake_prob_index', None)
        if fake_prob_index is not None and logits.dim() == 2 and logits.shape[1] > 1:
            logits = _prob_tensor_to_binary_logits(logits[:, fake_prob_index])
        elif logits.dim() == 1:
            logits = _prob_tensor_to_binary_logits(logits)
        elif logits.dim() == 2 and logits.shape[1] == 1:
            logits = _prob_tensor_to_binary_logits(logits.view(-1))
        try:
            probs_fake = torch.softmax(logits, dim=1)[:, 1]
            if should_log:
                probs_cpu = probs_fake.detach().cpu().tolist()
                if len(probs_cpu) == 1:
                    print(f'fake prob: {probs_cpu[0]:.6f}')
                else:
                    print(f'fake prob (per-sample): {[round(p, 6) for p in probs_cpu]}  mean={float(torch.tensor(probs_cpu).mean()):.6f}')
            # 输出 soft goal 提示（非硬性停止）：阈值默认 real<0.1, fake>0.9
            p_low = getattr(self.args, 'prob_low_real', 0.1)
            p_high = getattr(self.args, 'prob_high_fake', 0.9)
            if should_log:
                below = (probs_fake < p_low).sum().item()
                above = (probs_fake > p_high).sum().item()
                print(f'soft goals: prob<={p_low} count {below}, prob>={p_high} count {above}')
        except Exception as _:
            pass

        self.counter += 1
        return logits

    def recon(self, x):
        """返回重建图（[0,1]）。
        
        默认返回256×256（DIRE计算尺寸）。
        如果设置 recon_output_224=True，使用 CenterCrop 到 224 匹配分类器输入。
        """
        is_imagenet = ('imagenet' in getattr(self.args, 'domain', ''))
        need_224 = bool(getattr(self.args, 'recon_output_224', False))
        
        x_in = x
        if is_imagenet and x.shape[-1] != 256:
            print(f"[warn] recon: Input size {x.shape[-2:]} != 256, upsampling")
            x_in = F.interpolate(x_in, size=(256, 256), mode='bilinear', align_corners=False)

        x_m1_1 = (x_in - 0.5) * 2.0
        recon_m1_1 = self.recon_model.recon_from_x(x_m1_1, enable_grad=True)
        recon_01 = torch.clamp((recon_m1_1 + 1.0) * 0.5, 0.0, 1.0)

        # 如果需要224（与分类器输入一致），使用 TF.center_crop
        if is_imagenet and need_224:
            recon_01 = TF.center_crop(recon_01, 224)
        
        return recon_01
