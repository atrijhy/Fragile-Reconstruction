"""
Unified DIRE inference wrappers for attacks:
- DIRE_BPDA_Model: forward(x, mode) exposes two modes needed by BPDA_EOT_Attack:
	- mode='purify': returns DIRE-processed image in [0,1] (no grad expected by BPDA)
	- mode='classify': given an input in [0,1], returns classifier logits
  The internal reconstruction backend can be:
	- 'ddpm': use original GuidedDiffusion (non-differentiable)
	- 'ddim': use GuidedDiffusionDDIM (we still detach in BPDA mode to mimic non-diff forward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_image_classifier
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_guided_ddim import GuidedDiffusionDDIM


class DIRE_BPDA_Model(nn.Module):
	def __init__(self, args, config):
		super().__init__()
		self.args = args
		self.config = config

		# classifier on top of DIRE features (optionally load external checkpoint)
		self.classifier = get_image_classifier(
			args.classifier_name,
			ckpt_path=getattr(args, 'classifier_ckpt', None)
		).to(config.device)

		# choose reconstruction backend
		backend = getattr(args, 'dire_recon_backend', 'ddpm')
		self.backend = backend
		if backend == 'ddpm':
			self.runner = GuidedDiffusion(args, config, device=config.device)
		elif backend == 'ddim':
			self.runner = GuidedDiffusionDDIM(args, config, device=config.device)
		else:
			raise ValueError(f'Unknown dire_recon_backend: {backend}')

	def _recon_m1_1(self, x01: torch.Tensor) -> torch.Tensor:
		"""Reconstruct image via diffusion backend. Input/Output in [-1,1]."""
		x_m1_1 = (x01 - 0.5) * 2.0
		# Use official DIRE pathway: DDIM reverse to latent, then decode
		if isinstance(self.runner, GuidedDiffusionDDIM):
			# DDIM provides recon_from_x with deterministic reverse+decode
			x_re = self.runner.recon_from_x(x_m1_1, enable_grad=False)
		else:
			# DDPM backend: reconstruct via DDIM reverse + DDPM decode
			x_re = self.runner.recon_from_x(x_m1_1)
		return x_re

	def dire_map(self, x01: torch.Tensor) -> torch.Tensor:
		"""Compute DIRE feature |x - recon(x)|, return in [0,1]."""
		x_re_m1_1 = self._recon_m1_1(x01)
		dire_m1_1 = torch.abs((x01 - 0.5) * 2.0 - x_re_m1_1)
		# Optionally normalize magnitude to [0,1]
		# Here we simply clamp after rescale to [0,1]
		dire_01 = torch.clamp(dire_m1_1, 0.0, 2.0) * 0.5
		return dire_01

	def purify(self, x01: torch.Tensor) -> torch.Tensor:
		"""
		BPDA expects purify(x) -> x_purified in [0,1]. Here we output DIRE feature in [0,1].

		For ImageNet, keep parity with the e2e path and the official diffusion model's
		training resolution (256): upsample input 224->256 for reconstruction/DIRE,
		then downsample DIRE back to 224 for the classifier.
		"""
		with torch.no_grad():
			is_imagenet = ('imagenet' in getattr(self.args, 'domain', ''))
			x_in = x01
			if is_imagenet:
				# upsample to 256x256 before reconstruction (guided diffusion pretrained at 256)
				x_in = F.interpolate(x_in, size=(256, 256), mode='bilinear', align_corners=False)
			# compute DIRE in [0,1]
			dire_01 = self.dire_map(x_in)
			if is_imagenet:
				# bring back to 224x224 for downstream classifier
				dire_01 = F.interpolate(dire_01, size=(224, 224), mode='bilinear', align_corners=False)
			return dire_01

	def forward(self, x, mode: str = 'purify_and_classify'):
		if mode == 'purify':
			return self.purify(x)
		elif mode == 'classify':
			return self._classify_logits(x)
		elif mode == 'purify_and_classify':
			x_p = self.purify(x)
			return self._classify_logits(x_p)
		else:
			raise NotImplementedError(f'Unknown mode: {mode}')

	def _classify_logits(self, x: torch.Tensor) -> torch.Tensor:
		logits = self.classifier(x)
		# Align with DIRE_Adv_Model: convert binary outputs to 2-class logits [real,fake]
		fake_prob_index = getattr(self.args, 'fake_prob_index', None)
		if fake_prob_index is not None and logits.dim() == 2 and logits.shape[1] > 1:
			probs = logits[:, fake_prob_index]
			if torch.any(probs < 0) or torch.any(probs > 1):
				probs = torch.sigmoid(probs)
			probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
			logits = torch.log(torch.stack([1.0 - probs, probs], dim=1))
		elif logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
			probs = logits.view(-1)
			if torch.any(probs < 0) or torch.any(probs > 1):
				probs = torch.sigmoid(probs)
			probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
			logits = torch.log(torch.stack([1.0 - probs, probs], dim=1))
		if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
			probs = logits.view(-1)
			if torch.any(probs < 0) or torch.any(probs > 1):
				probs = torch.sigmoid(probs)
			probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
			logits = torch.log(torch.stack([1.0 - probs, probs], dim=1))
		return logits

