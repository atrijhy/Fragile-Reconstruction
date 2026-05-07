import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models import VQModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from joblib.hashing import hash
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.v2.functional import to_pil_image
from tqdm import tqdm

from aeroblade.data import ImageFolder, read_files
from aeroblade.inversion import create_pipeline
from aeroblade.misc import device, safe_mkdir, write_config


@torch.no_grad()
def compute_reconstructions(
    ds: Dataset,
    repo_id: str,
    output_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    iterations: int = 1,
    seed: int = 1,
    batch_size: int = 1,
    num_workers: int = 1,
    chunk_size: Optional[int] = None,
    device_str: Optional[str] = None,
    decode_device_str: Optional[str] = None,
    use_cache: bool = True,
    target_size: Optional[int] = None,
) -> list[Path]:
    """Compute AE reconstructions and save them in a unique directory."""
    if output_root is None and output_dir is None:
        raise ValueError("Either output_root or output_dir must be specified.")
    if output_root is not None and output_dir is not None:
        print("Ignoring output_root since output_dir is specified.")

    arg_dict = {"ds": ds, "repo_id": repo_id, "output_root": output_root, "seed": seed}
    if output_dir is None:
        # create output directory based on hashed arguments if not specified
        output_dir = output_root / hash(arg_dict) / str(iterations)

    # 重新计算 / 继续增量计算（是否真正重建由 use_cache+chunk 控制）
    safe_mkdir(output_dir)
    write_config(arg_dict, output_dir.parent)

    # if more than one iteration, recursively load previous iterations
    if iterations > 1:
        previous_paths = compute_reconstructions(
            ds=ds,
            repo_id=repo_id,
            output_root=output_root,
            iterations=iterations - 1,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        ds = ImageFolder(paths=previous_paths, transform=ds.transform, amount=ds.amount)

    # 仅加载 AE（VAE 或 MOVQ），避免加载整个 Stable Diffusion pipeline，显著降低显存占用
    encode_device = torch.device(device_str or device())
    decode_device = torch.device(decode_device_str or device_str or device())

    def load_autoencoder(target_device: torch.device):
        # GPU 上用 float16，CPU 上用 float32，避免 slow_conv2d_cpu 不支持 float16 的报错
        dtype = torch.float16 if target_device.type == "cuda" else torch.float32
        if "kandinsky-2" in repo_id:
            model = VQModel.from_pretrained(
                repo_id,
                subfolder="movq",
                torch_dtype=dtype,
            )
        else:
            model = AutoencoderKL.from_pretrained(
                repo_id,
                subfolder="vae",
                torch_dtype=dtype,
            )
        return model.to(target_device)

    ae_encode = load_autoencoder(encode_device)
    ae_decode = ae_encode if decode_device == encode_device else load_autoencoder(
        decode_device
    )
    encode_dtype = next(iter(ae_encode.parameters())).dtype
    decode_dtype = next(iter(ae_decode.parameters())).dtype

    generator = torch.Generator(device=encode_device).manual_seed(seed)
    total = len(ds)
    if total == 0:
        return []
    if chunk_size is None or chunk_size <= 0:
        chunk_size = total
    num_chunks = math.ceil(total / chunk_size)
    all_indices = list(range(total))

    # 尽量直接用 ImageFolder 暴露的 img_paths，避免额外读取；
    # 注意：目录中可能有“历史残留”的 PNG，我们后面会只按当前 ds 顺序返回路径，确保 1:1 对齐。
    ds_paths = getattr(ds, "img_paths", None)

    for chunk_idx, chunk_start in enumerate(range(0, total, chunk_size), start=1):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_indices = all_indices[chunk_start:chunk_end]

        # 从头完整重建：不再根据磁盘是否已有 PNG 跳过任何样本
        pending_indices = chunk_indices

        chunk_subset = Subset(ds, pending_indices)
        loader = DataLoader(
            chunk_subset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        chunk_desc = f"Reconstructing with {repo_id} (chunk {chunk_idx}/{num_chunks})"

        for images, paths in tqdm(loader, desc=chunk_desc):
            # 如果指定了 target_size，先 resize 图片（使用双线性插值）
            if target_size is not None and target_size > 0:
                images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
            # 归一化到 [-1, 1]
            images = images.to(encode_device, dtype=encode_dtype) * 2.0 - 1.0

            # encode
            latents = retrieve_latents(ae_encode.encode(images), generator=generator)

            # decode
            latents_dec = latents.to(decode_device, dtype=decode_dtype)
            if isinstance(ae_decode, VQModel):
                reconstructions = ae_decode.decode(
                    latents_dec,
                    force_not_quantize=True,
                    return_dict=False,
                )[0]
            else:
                reconstructions = ae_decode.decode(
                    latents_dec, return_dict=False
                )[0]

            # 反归一化回 [0,1]
            reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)

            # 保存
            for reconstruction, path in zip(reconstructions, paths):
                reconstruction_path = output_dir / f"{Path(path).stem}.png"
                to_pil_image(reconstruction).save(reconstruction_path)

            # 及时释放中间 tensor，减少显存碎片
            del images, latents, latents_dec, reconstructions
            if encode_device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"Images saved to {output_dir}.")

    # 按 ds 的顺序生成重建文件路径列表，确保与 ds 严格 1:1 对齐
    aligned_paths: list[Path] = []
    for idx in range(total):
        if ds_paths is not None:
            img_path = ds_paths[idx]
        else:
            _, img_path = ds[idx]
        aligned_paths.append(output_dir / f"{Path(img_path).stem}.png")
    return aligned_paths
    return reconstruction_paths


@torch.no_grad()
def compute_deeper_reconstructions(
    ds: Dataset,
    repo_id: str,
    output_root: Path,
    num_inference_steps: int,
    num_reconstruction_steps: int,
) -> list[Path]:
    """
    Compute reconstructions with AE and some inversion steps and save them in a
    unique directory.
    """
    # create output directory based on hashed arguments
    arg_dict = {
        "ds": ds,
        "repo_id": repo_id,
        "output_root": output_root,
        "num_inference_steps": num_inference_steps,
        "num_reconstruction_steps": num_reconstruction_steps,
    }
    output_dir = output_root / hash(arg_dict)

    # load files if output directory already exists, compute otherwise
    if not (
        output_dir.exists()
        and len(reconstruction_paths := read_files(output_dir)) == len(ds)
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_config(arg_dict, output_dir.parent)

        # set up pipeline
        pipe = create_pipeline(sd_model_ckpt=repo_id, use_blip_only=True)

        # reconstruct
        reconstruction_paths = []
        for _, paths in tqdm(
            DataLoader(ds, batch_size=1),
            desc=f"Reconstructing with {repo_id}.",
        ):
            path = paths[0]
            img = Image.open(path).convert("RGB").resize((512, 512))
            rec = pipe.compute_reconstruction(
                img,
                reconstruction_steps=num_reconstruction_steps,
                num_inference_steps=num_inference_steps,
            )

            # save
            reconstruction_path = output_dir / f"{Path(path).stem}.png"
            rec.save(reconstruction_path)
            reconstruction_paths.append(reconstruction_path)
        print(f"Images saved to {output_dir}.")
    return reconstruction_paths


def extract_patches(
    array: np.ndarray | torch.Tensor, size: int, stride: int
) -> np.ndarray | torch.Tensor:
    """
    Split 4D tensor into (overlapping) spatial patches.
    Output shape is batch_size x num_patches x num_channels x patch_size x patch_size
    """
    if isinstance(array, np.ndarray):
        is_ndarray = True
        array = torch.from_numpy(array)
    else:
        is_ndarray = False
    if array.ndim != 4:
        raise ValueError("array must be 4D.")
    patches = F.unfold(array, kernel_size=size, stride=stride).mT.reshape(
        array.shape[0], -1, array.shape[1], size, size
    )
    if is_ndarray:
        patches = patches.numpy()
    return patches
