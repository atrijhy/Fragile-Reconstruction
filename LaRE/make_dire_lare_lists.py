#!/usr/bin/env python3
"""
Generate LaRE annotation list files from a data directory.

Expected data structure:
  <data_root>/
    real/{train,val,test}/...
    adm/{train,val,test}/...
    biggan/{train,val,test}/...
    flux/{train,val,test}/...
    glide/{train,val,test}/...
    sdv5/{train,val,test}/...
    vqdm/{train,val,test}/...

Output example:
  LaRE/anns_dire/train_adm.txt   (absolute path + label)
    /data/adm/train/...png 1
    /data/real/train/...png 0

Pass these txt files to --train_file / --val_file in the LaRE training scripts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List


DEFAULT_GENS = ["adm", "biggan", "flux", "glide", "sdv5", "vqdm"]


def iter_images(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not root.exists():
        return []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def write_list(
    out_path: Path,
    real_split_dir: Path,
    fake_split_dir: Path,
) -> None:
    lines: List[str] = []
    # fake: label 1
    for p in sorted(iter_images(fake_split_dir)):
        lines.append(f"{p.resolve()} 1\n")
    # real: label 0
    for p in sorted(iter_images(real_split_dir)):
        lines.append(f"{p.resolve()} 0\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[write] {out_path}  (fake={fake_split_dir}, real={real_split_dir}, n={len(lines)})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate LaRE annotation txt files (absolute path + binary label)."
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("DATA_ROOT", "/path/to/data"),
        help="Root directory containing real/ and <gen>/ subdirectories (or set DATA_ROOT env var).",
    )
    ap.add_argument(
        "--gens",
        type=str,
        default="adm,biggan,flux,glide,sdv5,vqdm",
        help="Comma-separated generator names (subdirectories under data_root).",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated split names (typically train,val,test).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="anns_dire",
        help="Output directory for txt files (relative to LaRE root).",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    gens = [g.strip() for g in args.gens.split(",") if g.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    lare_root = Path(__file__).resolve().parent
    out_root = lare_root / args.out_dir

    real_root = data_root / "real"

    for gen in gens:
        fake_root = data_root / gen
        for split in splits:
            real_split = real_root / split
            fake_split = fake_root / split
            if not fake_split.exists():
                print(f"[skip] fake split not found: {fake_split}")
                continue
            if not real_split.exists():
                print(f"[skip] real split not found: {real_split}")
                continue
            out_name = f"{split}_{gen}.txt"
            out_path = out_root / out_name
            write_list(out_path, real_split, fake_split)


if __name__ == "__main__":
    main()

