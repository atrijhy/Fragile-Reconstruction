import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy and PyTorch for more deterministic runs.

    Note: full bitwise reproducibility is not guaranteed for operations like
    ODE solvers or non-deterministic CUDA kernels, but this aligns RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn flags
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
