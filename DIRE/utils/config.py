import argparse
import os
import sys
from abc import ABC
from typing import Type


class DefaultConfigs(ABC):
    ####### base setting ######
    gpus = [0]
    seed = 3407
    arch = "resnet50"
    datasets = ["zhaolian_train"]
    datasets_test = ["adm_res_abs_ddim20s"]
    mode = "binary"
    class_bal = False
    batch_size = 64
    loadSize = 256
    cropSize = 256  # ← 关键：保持 256 与离线/在线一致；224 裁剪由 dire_target_size 控制（仅分类器入口）
    epoch = "latest"
    num_workers = 20
    serial_batches = False
    isTrain = True

    # data augmentation
    rz_interp = ["bilinear"]
    blur_prob = 0.0
    blur_sig = [0.5]
    jpg_prob = 0.0
    jpg_method = ["cv2"]
    jpg_qual = [75]
    gray_prob = 0.0
    aug_resize = True
    aug_crop = True
    aug_flip = True
    aug_norm = True

    ####### train setting ######
    warmup = False
    warmup_epoch = 3
    earlystop = True
    earlystop_epoch = 5
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 20
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    nepoch = 400
    beta1 = 0.9
    lr = 0.0001

    # SGD hyper-parameters (only used when optim == 'sgd')
    sgd_momentum = 0.9
    sgd_weight_decay = 5e-4
    # AdamW hyper-parameters (only used when optim == 'adamw')
    adamw_weight_decay = 1e-2
    init_type = "normal"
    init_gain = 0.02
    pretrained = True

    # Adversarial training (FreeLB-style, applied in input space)
    # Disabled by default; enable via CLI opts: freelb_enable True
    freelb_enable = False
    # "freelb": original FreeLB-style (accumulate param grads across K inner steps)
    # "pgd":    standard PGD adversarial training (generate K-step adversary, then 1 backward on adv)
    freelb_method = "freelb"
    freelb_k = 3                 # inner steps per batch
    freelb_eps = 8 / 255         # perturbation radius (in the SAME space as model input)
    freelb_alpha = 2 / 255       # step size
    freelb_norm = "linf"         # "linf" | "l2"
    freelb_rand_init = True      # random init within eps ball
    freelb_warmup_steps = 0      # first N optimizer steps use clean training
    freelb_clean_weight = 1.0    # weight of clean loss in mix
    freelb_adv_weight = 1.0      # weight of adversarial loss in mix
    freelb_adv_prob = 1.0        # probability to apply adversarial training per batch (after warmup)
    freelb_log_delta_every = 0   # if >0, log |delta| stats every N steps (for debugging)

    # True PGD-AT (Madry et al.): find K-step adversarial, then single backward & update
    # DIFFERENT from FreeLB (which accumulates grads during K inner steps)
    pgd_enable = False
    pgd_k = 8                    # inner PGD steps
    pgd_eps = 4 / 255            # perturbation radius (in model input space)
    pgd_alpha = 1 / 255          # step size per iteration
    pgd_norm = "linf"            # "linf" | "l2"
    pgd_rand_init = True         # random init within eps ball
    pgd_warmup_steps = 0         # first N optimizer steps use clean training
    pgd_clean_weight = 0.0       # weight of clean loss (0 = pure adversarial training)
    pgd_adv_weight = 1.0         # weight of adversarial loss
    pgd_adv_loss = "bce"         # "bce" (standard PGD-AT) or "kl" (TRADES-style consistency)
    pgd_kl_beta = 1.0            # KL loss multiplier (only used when pgd_adv_loss=kl)
    pgd_adv_prob = 1.0           # probability to apply PGD per batch (after warmup)
    pgd_log_delta_every = 0      # if >0, log |delta| stats every N steps
    pgd_disable_clamp = False    # if True, disable input clamping (allow out-of-range adversarial inputs)

    # Input-space PGD: data in [0,1], preprocess AFTER perturbation
    # This allows eps/alpha to always be in [0,1] scale (e.g., 8/255)
    # Requires dire_raw_input=True in dataloader
    pgd_input_space = False              # enable input-space PGD mode
    pgd_preprocess_scale_to_m11 = False  # apply scale_to_m11 after perturbation
    pgd_preprocess_imagenet_norm = False # apply ImageNet norm after perturbation

    # torchattacks integration (alternative to manual PGD implementation)
    use_torchattacks = False             # use torchattacks library instead of manual PGD
    ta_eps = 8 / 255                     # torchattacks eps (same scale as pgd_eps)
    ta_alpha = 2 / 255                   # torchattacks step size
    ta_steps = 10                        # torchattacks number of steps

    # DIRE-specific settings
    dire_mode = False  # enable ODE-based DIRE pipeline
    dire_config_path = ""
    dire_diffusion_model_path = ""
    # Align training default to export/eval ODE settings
    dire_t_steps = 50
    dire_ode_method = "euler"
    dire_ode_step_size = 2e-2
    dire_ode_rtol = 1e-3
    dire_ode_atol = 1e-3
    dire_eot_reps = 1
    dire_fix_rand = True
    dire_smooth_eps = 5e-5
    dire_cache_maps = False  # caching breaks gradient, keep False for attacks
    dire_amp = False
    dire_train_real = ""
    dire_train_fake = ""
    dire_test_real = ""
    dire_test_fake = ""
    # optional explicit validation dirs (if provided, validation will use these instead of test dirs)
    dire_val_real = ""
    dire_val_fake = ""
    dire_target_size = 224  # ← 修改默认为 224，在送分类器前裁剪（与离线/在线对齐）
    # 当为 True 时，先将输入（无论 .pt/.png）统一到 256×256 再裁 target_size；
    # 当为 False 时，保持原生分辨率（例如 64×64），直接在该分辨率下做随机/中心裁与归一化。
    dire_base_256 = True
    dire_scale_to_m11 = False  # 默认值为 True，表示输入会被归一化到 [-1, 1]，建议保持一致性
    dire_apply_imagenet_norm = True  # when True, ignore scale_to_m11 and apply ImageNet normalization
    # Raw input mode: keep tensor in [0, 1] range for input-space PGD
    # When True, ignores scale_to_m11 and apply_imagenet_norm in dataloader
    dire_raw_input = False
    # DIRE map normalization for adversarial training
    # When True, DIRE maps are normalized from [0, dire_global_max] to [0, 1]
    dire_normalize = False
    dire_global_max = 0.2  # approximate max value of DIRE maps
    # augmentation for dire_mode training
    dire_aug_crop224 = True  # RandomCrop(224) during training
    dire_aug_flip = True     # RandomHorizontalFlip during training
    dire_flip_prob = 0.5
    # interpolation for dire_mode pipelines (Resize & tensor upsample when needed)
    # choices: 'bilinear' | 'bicubic' | 'lanczos' | 'nearest'
    dire_interp = 'bilinear'
    # 限制每类样本数量（0 表示不限制），分别控制训练集和验证集
    dire_max_train_samples = 0
    dire_max_val_samples = 0

    # K-fold cross-validation options (for dire_mode only)
    # When enabled, training uses train folds and validation uses the held-out fold from training dirs
    dire_use_kfold_val = False
    dire_kfold_k = 5
    dire_kfold_idx = 0  # which fold is used as validation [0..k-1]
    dire_kfold_seed = 3407

    # paths information
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_root = os.path.join(root_dir, "data")
    exp_root = os.path.join(root_dir, "data", "exp")
    _exp_name = ""
    exp_dir = ""
    ckpt_dir = ""
    logs_path = ""
    ckpt_path = ""

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter
    def exp_name(self, value: str):
        self._exp_name = value
        self.exp_dir: str = os.path.join(self.exp_root, self.exp_name)
        self.ckpt_dir: str = os.path.join(self.exp_dir, "ckpt")
        self.logs_path: str = os.path.join(self.exp_dir, "logs.txt")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def to_dict(self):
        dic = {}
        for fieldkey in dir(self):
            fieldvalue = getattr(self, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvalue) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvalue
        return dic


def args_list2dict(arg_list: list):
    assert len(arg_list) % 2 == 0, f"Override list has odd length: {arg_list}; it must be a list of pairs"
    return dict(zip(arg_list[::2], arg_list[1::2]))


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "yes", "on", "y", "t", "1"):
        return True
    elif v.lower() in ("false", "no", "off", "n", "f", "0"):
        return False
    else:
        return bool(v)


def str2list(v: str, element_type=None) -> list:
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


CONFIGCLASS = Type[DefaultConfigs]

parser = argparse.ArgumentParser()
# Make --gpus optional (default None). If not provided, keep cfg.gpus from the imported config
parser.add_argument("--gpus", default=None, type=int, nargs="+")
parser.add_argument("--exp_name", default="", type=str)
parser.add_argument("--ckpt", default="model_epoch_latest.pth", type=str)
parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
args = parser.parse_args()

if os.path.exists(os.path.join(DefaultConfigs.exp_root, args.exp_name, "config.py")):
    sys.path.insert(0, os.path.join(DefaultConfigs.exp_root, args.exp_name))
    from config import cfg

    cfg: CONFIGCLASS
else:
    cfg = DefaultConfigs()

if args.opts:
    opts = args_list2dict(args.opts)
    for k, v in opts.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unrecognized option: {k}")
        original_type = type(getattr(cfg, k))
        if original_type == bool:
            setattr(cfg, k, str2bool(v))
        elif original_type in (list, tuple, set):
            setattr(cfg, k, str2list(v, type(getattr(cfg, k)[0])))
        else:
            setattr(cfg, k, original_type(v))

if args.gpus is not None:
    cfg.gpus: list = args.gpus
else:
    cfg.gpus = getattr(cfg, "gpus", [0])
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in cfg.gpus])
# Only override exp_name if the user provided one on the command line. This avoids
# clobbering a per-experiment config imported from data/exp/<exp_name>/config.py
if args.exp_name:
    cfg.exp_name = args.exp_name

# Fallback: ensure exp_dir and logs_path are initialized. Some per-experiment
# config modules may not have set exp_name via the property setter, leaving
# cfg.logs_path empty which causes Logger.open to fail. If logs_path is empty,
# force a safe exp_name and recreate directories.
if not getattr(cfg, "logs_path", None):
    # prefer an explicit args.exp_name if provided, else use a generic name
    fallback_name = args.exp_name or getattr(cfg, "_exp_name", "exp_default")
    cfg.exp_name = fallback_name
cfg.ckpt_path: str = os.path.join(cfg.ckpt_dir, args.ckpt)

if isinstance(cfg.datasets, str):
    cfg.datasets = cfg.datasets.split(",")
