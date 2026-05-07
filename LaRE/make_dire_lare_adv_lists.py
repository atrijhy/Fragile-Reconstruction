#!/usr/bin/env python3
"""
Generate LaRE input lists for DIRE adversarial samples (AutoAttack APGD-CE).

Scans a directory of adversarial PNGs and produces:

  LaRE/anns_dire/adv_<gen>.txt

Format:
  /abs/path/to/sample_000000_adv.png 1
  /abs/path/to/sample_000001_adv.png 1
  ...
All adversarial samples are labelled fake=1.

Pass --adv-root to point at the directory containing
full_dire_resnet50_<gen>_test/ subdirectories.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List


DEFAULT_GENS = ["adm", "biggan", "flux", "glide", "sdv5", "vqdm"]


def iter_adv_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for p in sorted(root.glob("sample_*_adv.png")):
        if p.is_file():
            yield p


def write_list(out_path: Path, adv_dir: Path) -> None:
    lines: List[str] = []
    for p in iter_adv_images(adv_dir):
        lines.append(f"{p.resolve()} 1\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[write] {out_path}  (adv_dir={adv_dir}, n={len(lines)})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate LaRE input lists adv_<gen>.txt for adversarial samples (path + label 1)."
    )
    ap.add_argument(
        "--gens",
        type=str,
        default="adm,biggan,flux,glide,sdv5,vqdm",
        help="Comma-separated generator names (corresponding to full_dire_resnet50_<gen>_test subdirs).",
    )
    ap.add_argument(
        "--adv-root",
        type=str,
        default=os.environ.get("DIRE_ADV_ROOT", "/path/to/adv_images"),
        help="Root directory containing full_dire_resnet50_<gen>_test/ subdirs "
             "(or set DIRE_ADV_ROOT env var).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="anns_dire",
        help="Output directory for adv_<gen>.txt files (relative to LaRE root).",
    )
    args = ap.parse_args()

    gens = [g.strip() for g in args.gens.split(",") if g.strip()]

    lare_root = Path(__file__).resolve().parent
    out_root = lare_root / args.out_dir
    adv_root = Path(args.adv_root)

    for gen in gens:
        adv_dir = adv_root / f"full_dire_resnet50_{gen}_test"
        if not adv_dir.exists():
            print(f"[skip] adv dir not found: {adv_dir}")
            continue
        out_path = out_root / f"adv_{gen}.txt"
        write_list(out_path, adv_dir)


if __name__ == "__main__":
    main()
