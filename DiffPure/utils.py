# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import sys
import argparse
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from robustbench import load_model
import data


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name]=v

    return new_state_dict


# ------------------------------------------------------------------------
def get_accuracy(model, x_orig, y_orig, bs=64, device=torch.device('cuda:0')):
    n_batches = x_orig.shape[0] // bs
    acc = 0.
    for counter in range(n_batches):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        output = model(x)
        acc += (output.max(1)[1] == y).float().sum()

    return (acc / x_orig.shape[0]).item()


def get_image_classifier(classifier_name, ckpt_path: str = None):
    class _Wrapper_ResNet(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet
            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(3, 1, 1)
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(3, 1, 1)

        def forward(self, x):
            x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
            return self.resnet(x)

    # Prefer specific handlers before broad substrings
    if classifier_name.startswith('dire_resnet'):
        # Load DIRE binary ResNet (1-logit) and wrap with the SAME input normalization used in training.
        # By default this factory used ImageNet normalization. However, the DIRE training in this repo
        # typically fed DIRE maps scaled to [-1,1] (no ImageNet mean/std). To avoid a distribution mismatch,
        # you can request the training-aligned mode by naming the classifier with suffix "_m11",
        # e.g. --classifier_name dire_resnet50_m11. Otherwise, ImageNet normalization remains the default.
        # Expected usage: --classifier_name dire_resnet50[_m11] --classifier_ckpt <path-to-model_epoch_*.pth>
        import importlib.util as _ilu
        import os as _os, sys as _sys

        # Locate DIRE utils.get_network and ensure 'networks' package is importable
        dire_root = _os.path.realpath(_os.path.join(_os.path.dirname(__file__), '..', 'DIRE'))
        if dire_root not in _sys.path:
            _sys.path.insert(0, dire_root)
        utils_py = _os.path.realpath(_os.path.join(dire_root, 'utils', 'utils.py'))
        spec = _ilu.spec_from_file_location('dire_demo_utils', utils_py)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load get_network from {utils_py}')
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_network = getattr(mod, 'get_network')

        # Pick backbone from name (resnet50/resnet101), default resnet50
        arch = 'resnet50'
        if 'resnet101' in classifier_name:
            arch = 'resnet101'
        # Build a plain ResNet(num_classes=1). We'll load trained weights next.
        model = get_network(arch)  # num_classes=1, no wrapper here

        # Heuristic ckpt discovery if not provided
        def _find_dire_ckpt(_base_dir: str) -> str | None:
            preferred = None
            fallback = None
            for root, dirs, files in _os.walk(_base_dir):
                if _os.path.basename(root) == 'ckpt':
                    best = _os.path.join(root, 'model_epoch_best.pth')
                    latest = _os.path.join(root, 'model_epoch_latest.pth')
                    if _os.path.isfile(best):
                        if 'fold4' in root and preferred is None:
                            preferred = best
                        elif fallback is None:
                            fallback = best
                    elif _os.path.isfile(latest) and fallback is None:
                        fallback = latest
            return preferred or fallback

        if ckpt_path is None:
            base_dir = _os.path.realpath(_os.path.join(_os.path.dirname(__file__), '..', 'DIRE', 'data', 'exp'))
            guessed = _find_dire_ckpt(base_dir)
            if guessed is None:
                raise ValueError('dire_resnet requires --classifier_ckpt path to a trained checkpoint (.pth)')
            ckpt_path = guessed
            print(f"[dire_resnet] auto-selected ckpt: {ckpt_path}")

        sd = torch.load(ckpt_path, map_location='cpu')
        # Unwrap common training checkpoint format { 'model': state_dict, ... }
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        # If weights were saved from DireOdeResNet(wrapper), keys are prefixed with 'base.'.
        # Also tolerate 'module.' from DataParallel. Strip these to match a plain ResNet.
        if isinstance(sd, dict):
            def _strip_prefix(d, prefix):
                return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items() }
            if any(k.startswith('module.') for k in sd.keys()):
                sd = _strip_prefix(sd, 'module.')
            if any(k.startswith('base.') for k in sd.keys()):
                sd = _strip_prefix(sd, 'base.')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) or len(unexpected):
            print(f"[dire_resnet] loaded with missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()

        # Two input modes for compatibility with training:
        # - [-1,1] scaling (matches typical DIRE training pipeline) <-- now the default
        # - ImageNet normalization (opt-in via explicit '_imnet' suffix)
        class _Wrapper_Logit_IMNET(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.mu = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                self.sigma = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            def forward(self, x):
                x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
                out = self.m(x)
                return out.view(-1)

        class _Wrapper_Logit_M11(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                # Expect input in [0,1] and convert to [-1,1] to mimic training
                x = x * 2.0 - 1.0
                out = self.m(x)
                return out.view(-1)

        class _Wrapper_Logit_Default(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                # Default: expect input in [0,1] with no further normalization
                out = self.m(x)
                return out.view(-1)

        if '_imnet' in classifier_name:
            print(f'using DIRE {arch} (1-logit) with ImageNet normalization (explicit _imnet suffix)')
            wrapper_resnet = _Wrapper_Logit_IMNET(model)
        elif '_m11' in classifier_name:
            # default to training-aligned M11 for consistency
            print(f'using DIRE {arch} (1-logit) with [-1,1] input (training-aligned, default)')
            wrapper_resnet = _Wrapper_Logit_M11(model)
        else:
            print(f'using DIRE {arch} (1-logit) with [0,1] input (default)')
            wrapper_resnet = _Wrapper_Logit_Default(model)

    elif 'imagenet_adm' in classifier_name:
        # Mirror DIRE/DIRE/demo.py calling convention: ResNet50 -> 1-logit output; apply ImageNet normalization.
        if ckpt_path is None:
            raise ValueError('imagenet_adm requires --classifier_ckpt pointing to your imagenet_adm.pth')

        import importlib.util as _ilu
        import os as _os, sys as _sys
        # Add DIRE/DIRE to sys.path so that 'networks' package inside demo can be imported
        dire_root = _os.path.realpath(_os.path.join(_os.path.dirname(__file__), '..', 'DIRE'))
        if dire_root not in _sys.path:
            _sys.path.insert(0, dire_root)
        utils_py = _os.path.realpath(_os.path.join(dire_root, 'utils', 'utils.py'))
        spec = _ilu.spec_from_file_location('dire_demo_utils', utils_py)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load get_network from {utils_py}')
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_network = getattr(mod, 'get_network')
        print('using imagenet_adm (ResNet50 1-logit) with ImageNet normalization...')
        model = get_network('resnet50')  # num_classes=1
        sd = torch.load(ckpt_path, map_location='cpu')
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) or len(unexpected):
            print(f"[imagenet_adm] loaded with missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()

        class _Wrapper_Logit(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.mu = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                self.sigma = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            def forward(self, x):
                x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
                out = self.m(x)  # shape (N,1)
                return out.view(-1)  # 1-D logit vector

        wrapper_resnet = _Wrapper_Logit(model)

    elif 'binary_prob_torchscript' in classifier_name:
        # A generic TorchScript binary detector that outputs a single probability or logit for "fake".
        # Expect ckpt_path to point to a .pt/.ts TorchScript file. Input is assumed in [0,1].
        if ckpt_path is None:
            raise ValueError('binary_prob_torchscript requires ckpt_path to a TorchScript file')

        print(f'loading TorchScript binary probability model from {ckpt_path}...')
        ts_model = torch.jit.load(ckpt_path, map_location='cpu').eval()

        class _Wrapper_BinaryProb(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x)
                # Flatten common shapes to (N,) representing fake probability/logit
                if out.dim() > 1:
                    out = out.view(out.shape[0], -1)[:, 0]
                out = out.squeeze()
                # If output not in [0,1], treat it as logit and apply sigmoid
                if torch.any(out < 0) or torch.any(out > 1):
                    out = torch.sigmoid(out)
                return out

        wrapper_resnet = _Wrapper_BinaryProb(ts_model)

    elif 'imagenet' in classifier_name:
        if 'resnet18' in classifier_name:
            print('using imagenet resnet18...')
            model = models.resnet18(pretrained=True).eval()
        elif 'resnet50' in classifier_name:
            print('using imagenet resnet50...')
            model = models.resnet50(pretrained=True).eval()
        elif 'resnet101' in classifier_name:
            print('using imagenet resnet101...')
            model = models.resnet101(pretrained=True).eval()
        elif 'wideresnet-50-2' in classifier_name:
            print('using imagenet wideresnet-50-2...')
            model = models.wide_resnet50_2(pretrained=True).eval()
        elif 'deit-s' in classifier_name:
            print('using imagenet deit-s...')
            model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).eval()
        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = _Wrapper_ResNet(model)

    elif 'cifar10' in classifier_name:
        if 'wideresnet-28-10' in classifier_name:
            print('using cifar10 wideresnet-28-10...')
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-28-10-at0' in classifier_name:
            print('using cifar10 wrn-28-10-at0...')
            model = load_model(model_name='Gowal2021Improving_28_10_ddpm_100m', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-28-10-at1' in classifier_name:
            print('using cifar10 wrn-28-10-at1...')
            model = load_model(model_name='Gowal2020Uncovering_28_10_extra', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-at0' in classifier_name:
            print('using cifar10 wrn-70-16-at0...')
            model = load_model(model_name='Gowal2021Improving_70_16_ddpm_100m', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-at1' in classifier_name:
            print('using cifar10 wrn-70-16-at1...')
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-L2-at1' in classifier_name:
            print('using cifar10 wrn-70-16-L2-at1...')
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_model='L2')  # pixel in [0, 1]

        elif 'wideresnet-70-16' in classifier_name:
            print('using cifar10 wideresnet-70-16 (dm_wrn-70-16)...')
            from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish
            model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish)  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wresnet-76-10/weights-best.pt'
            print(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path)['model_state_dict']))
            model.eval()
            print(f"=> loaded wideresnet-70-16 checkpoint")

        elif 'resnet-50' in classifier_name:
            print('using cifar10 resnet-50...')
            from classifiers.cifar10_resnet import ResNet50
            model = ResNet50()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/resnet-50/weights.pt'
            print(f"=> loading resnet-50 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded resnet-50 checkpoint")

        elif 'wrn-70-16-dropout' in classifier_name:
            print('using cifar10 wrn-70-16-dropout (standard wrn-70-16-dropout)...')
            from classifiers.cifar10_resnet import WideResNet_70_16_dropout
            model = WideResNet_70_16_dropout()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wrn-70-16-dropout/weights.pt'
            print(f"=> loading wrn-70-16-dropout checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded wrn-70-16-dropout checkpoint")

        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = model

    elif 'celebahq' in classifier_name:
        attribute = classifier_name.split('__')[-1]  # `celebahq__Smiling`
        ckpt_path = f'pretrained/celebahq/{attribute}/net_best.pth'
        from classifiers.attribute_classifier import ClassifierWrapper
        model = ClassifierWrapper(attribute, ckpt_path=ckpt_path)
        wrapper_resnet = model
    elif 'imagenet_adm' in classifier_name:
        # Mirror DIRE/DIRE/demo.py calling convention: ResNet50 -> 1-logit output; apply ImageNet normalization.
        # ckpt_path should point to imagenet_adm.pth (or similar) as in your demo.
        if ckpt_path is None:
            raise ValueError('imagenet_adm requires --classifier_ckpt pointing to your imagenet_adm.pth')

        import importlib.util as _ilu
        import os as _os
        utils_py = _os.path.realpath(_os.path.join(_os.path.dirname(__file__), '..', 'DIRE', 'utils', 'utils.py'))
        spec = _ilu.spec_from_file_location('dire_demo_utils', utils_py)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load get_network from {utils_py}')
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_network = getattr(mod, 'get_network')
        print('using imagenet_adm (ResNet50 1-logit) with ImageNet normalization...')
        model = get_network('resnet50')  # num_classes=1
        sd = torch.load(ckpt_path, map_location='cpu')
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) or len(unexpected):
            print(f"[imagenet_adm] loaded with missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()

        class _Wrapper_Logit(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.mu = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                self.sigma = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            def forward(self, x):
                x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
                out = self.m(x)  # shape (N,1)
                return out.view(-1)  # 1-D logit vector

        wrapper_resnet = _Wrapper_Logit(model)
    elif 'binary_prob_torchscript' in classifier_name:
        # A generic TorchScript binary detector that outputs a single probability or logit for "fake".
        # Expect ckpt_path to point to a .pt/.ts TorchScript file. Input is assumed in [0,1].
        if ckpt_path is None:
            raise ValueError('binary_prob_torchscript requires ckpt_path to a TorchScript file')

        print(f'loading TorchScript binary probability model from {ckpt_path}...')
        ts_model = torch.jit.load(ckpt_path, map_location='cpu').eval()

        class _Wrapper_BinaryProb(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x)
                # Flatten common shapes to (N,) representing fake probability/logit
                if out.dim() > 1:
                    out = out.view(out.shape[0], -1)[:, 0]
                out = out.squeeze()
                # If output not in [0,1], treat it as logit and apply sigmoid
                if torch.any(out < 0) or torch.any(out > 1):
                    out = torch.sigmoid(out)
                return out

        wrapper_resnet = _Wrapper_BinaryProb(ts_model)
    else:
        raise NotImplementedError(f'unknown {classifier_name}')

    # Optionally load external checkpoint (state_dict) for non-TorchScript models
    if ckpt_path is not None and all(k not in classifier_name for k in ['binary_prob_torchscript', 'imagenet_adm', 'dire_resnet']):
        if os.path.isfile(ckpt_path):
            try:
                sd = torch.load(ckpt_path, map_location='cpu')
                # Allow either full state_dict or wrapped in dict
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                missing, unexpected = wrapper_resnet.load_state_dict(sd, strict=False)
                print(f"[classifier_ckpt] Loaded '{ckpt_path}' (missing={len(missing)}, unexpected={len(unexpected)})")
            except Exception as e:
                print(f"[classifier_ckpt] Failed to load {ckpt_path}: {e}")
        else:
            print(f"[classifier_ckpt] File not found: {ckpt_path}")
    return wrapper_resnet


class _FolderImageDataset(Dataset):
    def __init__(self, file_label_pairs, transform=None):
        self.samples = file_label_pairs
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # return filename (basename) for saving adversarial images consistently
        return img, label, os.path.basename(path)


def _collate_with_names(batch):
    """Collate (img, label, name) -> stacked imgs, tensor labels, list names."""
    imgs, labels, names = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    names = list(names)
    return imgs, labels, names

def _gather_image_files(root_dir, exts: set | None = None):
    """Recursively gather image files under root_dir using Path.rglob.

    Args:
        root_dir: str or Path
        exts: set of lowercase extensions including the leading dot (e.g. {'.png', '.jpg'})
    Returns:
        sorted list of file paths (Path objects)
    """
    if exts is None:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    root = Path(root_dir)
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)

def load_data(args, adv_batch_size):
    # ---- Custom hierarchical dataset path (real + optional fake) ----
    if getattr(args, 'data_root', None):
        # try to set deterministic seed same as export script
        try:
            from DIRE.utils.seed import set_seed
            set_seed(getattr(args, 'seed', 0))
        except Exception:
            pass

        data_root = args.data_root
        real_dir = os.path.join(data_root, args.real_subdir)
        assert os.path.isdir(real_dir), f'real_dir not found: {real_dir}'

        # Basic transforms: ensure consistent spatial size for batching (match export_dire_maps.py)
        if 'imagenet' in getattr(args, 'domain', ''):
            # Match export/eval pipeline resolution using args.dire_image_size (default 256)
            _sz = int(getattr(args, 'dire_image_size', 256) or 256)
            base_tf = transforms.Compose([
                transforms.Resize(_sz, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(_sz),
                transforms.ToTensor(),
            ])
        else:
            base_tf = transforms.Compose([transforms.ToTensor()])

        # extensions: honor args.glob_exts when present (match export_dire_maps.py)
        if getattr(args, 'glob_exts', None):
            exts = {e.strip().lower() for e in args.glob_exts.split(',') if e.strip()}
        else:
            exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

        file_label_pairs = []
        real_files = _gather_image_files(Path(real_dir), exts)
        real_offset = max(0, int(getattr(args, 'real_offset', 0) or 0))
        real_count = getattr(args, 'real_count', None)
        real_range_used = real_offset > 0 or real_count is not None
        if real_offset >= len(real_files):
            selected_real_files = []
        else:
            if real_count is not None and real_count >= 0:
                selected_real_files = real_files[real_offset:real_offset + real_count]
            else:
                selected_real_files = real_files[real_offset:]
        real_label = getattr(args, 'real_label', 0)
        if not getattr(args, 'only_fake', False):
            for p in selected_real_files:
                file_label_pairs.append((p, real_label))
        fake_range_used = False
        if getattr(args, 'fake_dirs', None):
            fake_label = getattr(args, 'fake_label', 1)
            fake_offset = max(0, int(getattr(args, 'fake_offset', 0) or 0))
            fake_count = getattr(args, 'fake_count', None)
            fake_range_used = fake_offset > 0 or fake_count is not None
            for d in args.fake_dirs.split(','):
                d = d.strip()
                if not d:
                    continue
                fake_path = os.path.join(data_root, d)
                if not os.path.isdir(fake_path):
                    print(f'[warn] fake dir not found: {fake_path}, skip.')
                    continue
                if not getattr(args, 'only_real', False):
                    fake_files = _gather_image_files(Path(fake_path), exts)
                    if fake_offset >= len(fake_files):
                        selected_fake_files = []
                    else:
                        if fake_count is not None and fake_count >= 0:
                            selected_fake_files = fake_files[fake_offset:fake_offset + fake_count]
                        else:
                            selected_fake_files = fake_files[fake_offset:]
                    for p in selected_fake_files:
                        file_label_pairs.append((p, fake_label))
        assert len(file_label_pairs) > 0, 'No images collected under data_root.'
        # Optional class-wise picking overrides num_sub when provided
        pick_real = getattr(args, 'pick_real', None)
        pick_fake = getattr(args, 'pick_fake', None)
        if pick_real is not None or pick_fake is not None:
            g = torch.Generator().manual_seed(getattr(args, 'data_seed', 0))
            reals = [p for p in file_label_pairs if p[1] == real_label]
            fakes = [p for p in file_label_pairs if p[1] != real_label]
            picked = []
            if pick_real is not None and len(reals) > 0:
                idx = torch.randperm(len(reals), generator=g)[:min(pick_real, len(reals))].tolist()
                picked.extend([reals[i] for i in idx])
            if pick_fake is not None and len(fakes) > 0:
                idx = torch.randperm(len(fakes), generator=g)[:min(pick_fake, len(fakes))].tolist()
                picked.extend([fakes[i] for i in idx])
            if len(picked) > 0:
                file_label_pairs = picked
        else:
            # Fallback: global num_sub
            if (not real_range_used and not fake_range_used
                    and getattr(args, 'num_sub', None) and args.num_sub > 0
                    and len(file_label_pairs) > args.num_sub):
                g = torch.Generator().manual_seed(getattr(args, 'data_seed', 0))
                perm = torch.randperm(len(file_label_pairs), generator=g)[:args.num_sub].tolist()
                file_label_pairs = [file_label_pairs[i] for i in perm]
        ds = _FolderImageDataset(file_label_pairs, transform=base_tf)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False, pin_memory=True, num_workers=4, collate_fn=_collate_with_names)
        batch = next(iter(loader))
        if len(batch) == 3:
            x_val, y_val, names = batch
        else:
            x_val, y_val = batch
            names = None
        print(f'[custom data_root] collected {len(ds)} images from real={real_dir} and optional fake_dirs={args.fake_dirs}')
        x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
        print(f'x_val shape: {x_val.shape}')
        # attach names to args for downstream saving
        setattr(args, 'sample_names', names)
        return x_val, y_val

    # ---- Original predefined domains ----
    if 'imagenet' in args.domain:
        val_dir = './dataset/imagenet_lmdb/val'  # using imagenet lmdb data
        val_transform = data.get_transform(args.domain, 'imval', base_size=224)
        val_data = data.imagenet_lmdb_dataset_sub(val_dir, transform=val_transform,
                                                  num_sub=args.num_sub, data_seed=args.data_seed)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    elif 'cifar10' in args.domain:
        data_dir = './dataset'
        transform = transforms.Compose([transforms.ToTensor()])
        val_data = data.cifar10_dataset_sub(data_dir, transform=transform,
                                            num_sub=args.num_sub, data_seed=args.data_seed)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    elif 'celebahq' in args.domain:
        data_dir = './dataset/celebahq'
        attribute = args.classifier_name.split('__')[-1]  # `celebahq__Smiling`
        val_transform = data.get_transform('celebahq', 'imval')
        clean_dset = data.get_dataset('celebahq', 'val', attribute, root=data_dir, transform=val_transform,
                                      fraction=2, data_seed=args.data_seed)  # data_seed randomizes here
        loader = DataLoader(clean_dset, batch_size=adv_batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(loader))  # [0, 1], 256x256
    else:
        raise NotImplementedError(f'Unknown domain: {args.domain}!')

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return x_val, y_val


# Streaming loader: avoid loading all samples into memory at once
def load_data_loader(args, batch_size: int):
    """Return a DataLoader yielding mini-batches with the same transforms as load_data.

    Only the custom data_root path is implemented for streaming. For builtin domains
    you can still use load_data (small subsets) or extend similarly.
    """
    if getattr(args, 'data_root', None):
        try:
            from DIRE.utils.seed import set_seed
            set_seed(getattr(args, 'seed', 0))
        except Exception:
            pass

        data_root = args.data_root
        real_dir = os.path.join(data_root, args.real_subdir)
        assert os.path.isdir(real_dir), f'real_dir not found: {real_dir}'

        # transforms match export/eval
        if 'imagenet' in getattr(args, 'domain', ''):
            _sz = int(getattr(args, 'dire_image_size', 256) or 256)
            base_tf = transforms.Compose([
                transforms.Resize(_sz, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(_sz),
                transforms.ToTensor(),
            ])
        else:
            base_tf = transforms.Compose([transforms.ToTensor()])

        # extensions
        if getattr(args, 'glob_exts', None):
            exts = {e.strip().lower() for e in args.glob_exts.split(',') if e.strip()}
        else:
            exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

        file_label_pairs = []
        real_files = _gather_image_files(Path(real_dir), exts)
        real_offset = max(0, int(getattr(args, 'real_offset', 0) or 0))
        real_count = getattr(args, 'real_count', None)
        if real_offset >= len(real_files):
            selected_real_files = []
        else:
            if real_count is not None and real_count >= 0:
                selected_real_files = real_files[real_offset:real_offset + real_count]
            else:
                selected_real_files = real_files[real_offset:]
        real_label = getattr(args, 'real_label', 0)
        if not getattr(args, 'only_fake', False):
            for p in selected_real_files:
                file_label_pairs.append((p, real_label))

        if getattr(args, 'fake_dirs', None):
            fake_label = getattr(args, 'fake_label', 1)
            fake_offset = max(0, int(getattr(args, 'fake_offset', 0) or 0))
            fake_count = getattr(args, 'fake_count', None)
            for d in args.fake_dirs.split(','):
                d = d.strip()
                if not d:
                    continue
                fake_path = os.path.join(data_root, d)
                if not os.path.isdir(fake_path):
                    print(f'[warn] fake dir not found: {fake_path}, skip.')
                    continue
                if not getattr(args, 'only_real', False):
                    fake_files = _gather_image_files(Path(fake_path), exts)
                    if fake_offset >= len(fake_files):
                        selected_fake_files = []
                    else:
                        if fake_count is not None and fake_count >= 0:
                            selected_fake_files = fake_files[fake_offset:fake_offset + fake_count]
                        else:
                            selected_fake_files = fake_files[fake_offset:]
                    for p in selected_fake_files:
                        file_label_pairs.append((p, fake_label))

        assert len(file_label_pairs) > 0, 'No images collected under data_root.'

        # Optional subsampling (same semantics as load_data)
        pick_real = getattr(args, 'pick_real', None)
        pick_fake = getattr(args, 'pick_fake', None)
        if pick_real is not None or pick_fake is not None:
            g = torch.Generator().manual_seed(getattr(args, 'data_seed', 0))
            reals = [p for p in file_label_pairs if p[1] == real_label]
            fakes = [p for p in file_label_pairs if p[1] != real_label]
            picked = []
            if pick_real is not None and len(reals) > 0:
                idx = torch.randperm(len(reals), generator=g)[:min(pick_real, len(reals))].tolist()
                picked.extend([reals[i] for i in idx])
            if pick_fake is not None and len(fakes) > 0:
                idx = torch.randperm(len(fakes), generator=g)[:min(pick_fake, len(fakes))].tolist()
                picked.extend([fakes[i] for i in idx])
            if len(picked) > 0:
                file_label_pairs = picked
        else:
            if getattr(args, 'num_sub', None) and args.num_sub > 0 and len(file_label_pairs) > args.num_sub:
                g = torch.Generator().manual_seed(getattr(args, 'data_seed', 0))
                perm = torch.randperm(len(file_label_pairs), generator=g)[:args.num_sub].tolist()
                file_label_pairs = [file_label_pairs[i] for i in perm]

        ds = _FolderImageDataset(file_label_pairs, transform=base_tf)
        num_workers = int(getattr(args, 'dataloader_workers', 4) or 4)
        loader = DataLoader(
            ds,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=_collate_with_names,
        )
        print(f"[stream] dataset size={len(ds)}, batch_size={batch_size}, workers={num_workers}")
        return loader, len(ds)

    raise NotImplementedError('Streaming loader is currently implemented only for custom --data_root path.')
