# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random

import torch
import torchvision.utils as tvu

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


class GuidedDiffusionDDIM(torch.nn.Module):
    def __init__(self, args, config, device=None, model_dir='pretrained/guided_diffusion'):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        
        # DDIM参数
        self.ddim_eta = getattr(args, 'ddim_eta', 0.0)  # 默认为确定性采样

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion(**model_config)
        # custom diffusion checkpoint if provided
        ckpt_path = getattr(args, 'diffusion_model_path', None)
        if ckpt_path is None:
            ckpt_path = f'{model_dir}/256x256_diffusion_uncond.pt'
        print(f'[diffusion] loading checkpoint: {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)

        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)

    def image_editing_sample(self, img, bs_id=0, tag=None, eta=None, enable_grad=False):
        """
        DDIM采样版本的image_editing_sample
        
        Args:
            img: 输入图像tensor
            bs_id: batch id，用于保存中间结果
            tag: 标识符
            eta: DDIM的随机性参数，0为完全确定性，1接近DDPM，None使用默认值
            enable_grad: 是否启用梯度计算（用于攻击）
        """
        if eta is None:
            eta = self.ddim_eta
            
        # 根据enable_grad决定是否使用torch.no_grad()
        if not enable_grad:
            return self._sample_with_no_grad(img, bs_id, tag, eta)
        else:
            return self._sample_with_grad(img, bs_id, tag, eta)
    
    def _sample_with_no_grad(self, img, bs_id, tag, eta):
        """不可微分版本：内存高效"""
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            batch_size = img.shape[0]

            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            img = img.to(self.device)
            x0 = img

            if bs_id < 2:
                os.makedirs(out_dir, exist_ok=True)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))

            xs = []
            for it in range(self.args.sample_step):
                # 添加噪声到指定的timestep
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                if bs_id < 2:
                    tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

                # 使用DDIM进行去噪
                x = self._ddim_denoising_loop(x, total_noise_levels, bs_id, out_dir, it, eta)

                x0 = x

                if bs_id < 2:
                    torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

                xs.append(x0)

            return torch.cat(xs, dim=0)
    
    def _sample_with_grad(self, img, bs_id, tag, eta):
        """可微分版本：支持梯度计算"""
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        if tag is None:
            tag = 'rnd' + str(random.randint(0, 10000))
        out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

        assert img.ndim == 4, img.ndim
        img = img.to(self.device)
        x0 = img
        xs = []
        for it in range(self.args.sample_step):
            # 添加噪声到指定的timestep
            e = torch.randn_like(x0, device=x0.device)
            total_noise_levels = self.args.t
            a = (1 - self.betas).cumprod(dim=0)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

            # 使用DDIM进行去噪（可微分版本）
            x = self._ddim_denoising_loop_grad(x, total_noise_levels, eta)
            x0 = x
            xs.append(x0)

        return torch.cat(xs, dim=0)
    
    def _ddim_denoising_loop(self, x, total_noise_levels, bs_id, out_dir, it, eta=0.0):
        """
        DDIM去噪循环
        """
        batch_size = x.shape[0]
        
        for i in reversed(range(total_noise_levels)):
            t = torch.tensor([i] * batch_size, device=self.device)

            x = self.diffusion.ddim_sample(
                self.model, 
                x, 
                t,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
                eta=eta
            )["sample"]

            # 保存中间步骤可视化
            if (i - 99) % 100 == 0 and bs_id < 2:
                tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'ddim_noise_t_{i}_{it}.png'))

        return x
    
    def _ddim_denoising_loop_grad(self, x, total_noise_levels, eta=0.0):
        """
        可微分的DDIM去噪循环（用于攻击）
        """
        batch_size = x.shape[0]
        
        for i in reversed(range(total_noise_levels)):
            t = torch.tensor([i] * batch_size, device=self.device)

            x = self.diffusion.ddim_sample(
                self.model, 
                x, 
                t,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
                eta=eta
            )["sample"]

        return x

    def _ddim_reverse_loop_grad(self, x, total_noise_levels):
        """
        可微分的DDIM反向（编码）循环：x_{t} = f_rev(x_{t-1})，从t=0到t=total_noise_levels-1。
        输入x应为[-1,1]域中的图像（x_0）。
        返回在指定步数后的潜变量x_T。
        """
        batch_size = x.shape[0]
        for i in range(0, int(total_noise_levels)):
            t = torch.tensor([i] * batch_size, device=self.device)
            x = self.diffusion.ddim_reverse_sample(
                self.model,
                x,
                t,
                clip_denoised=True,
                denoised_fn=None,
                model_kwargs=None,
                eta=0.0,  # Reverse ODE path必须确定性
            )["sample"]
        return x

    def recon_from_x(self, x, steps=None, eta=None, enable_grad=True):
        """
        可微分的DIRE重建：先编码（reverse），再解码（ddim）。
        - x: [-1,1]域图像
        - steps: 使用的时间步数（默认self.args.t）
        - eta: DDIM的eta参数（默认self.ddim_eta）
        - enable_grad: 是否启用梯度
        返回：reconstruction，形状与x相同
        """
        if steps is None:
            steps = int(self.args.t)
        if eta is None:
            eta = float(self.ddim_eta)

        if not enable_grad:
            # no-grad分支，节省显存
            with torch.no_grad():
                latent = self._ddim_reverse_loop_grad(x, steps)
                recon = self._ddim_denoising_loop_grad(latent, steps, eta)
                return recon
        # 可微分分支
        latent = self._ddim_reverse_loop_grad(x, steps)
        recon = self._ddim_denoising_loop_grad(latent, steps, eta)
        return recon

    def dire_map(self, x, steps=None, eta=None, enable_grad=True, normalize=True):
        """
        计算可微分的DIRE特征：|x - recon(x)|。
        - normalize=True时，将范围从[0,2]缩放到[0,1]（除以2）。
        返回：与x同形状的张量。
        """
        recon = self.recon_from_x(x, steps=steps, eta=eta, enable_grad=enable_grad)
        dire = torch.abs(x - recon)
        if normalize:
            dire = 0.5 * dire
        return dire
    
    def image_editing_sample_deterministic(self, img, bs_id=0, tag=None):
        """
        完全确定性的DDIM采样（eta=0.0）
        """
        return self.image_editing_sample(img, bs_id, tag, eta=0.0)
    
    def image_editing_sample_stochastic(self, img, bs_id=0, tag=None, eta=1.0):
        """
        随机性DDIM采样（eta接近1.0，类似DDPM）
        """
        return self.image_editing_sample(img, bs_id, tag, eta=eta)
    
    def forward(self, x):
        """
        前向传播方法，用于攻击时计算梯度
        """
        return self.image_editing_sample(x, bs_id=0, tag=None, eta=0.0, enable_grad=True)
