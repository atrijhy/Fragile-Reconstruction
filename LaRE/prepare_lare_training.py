#!/usr/bin/env python3
"""
Prepare the map_file (LaRE feature list) required for LaRE2 training.

Prerequisites:
  1) Generate annotation lists from your data directory:
       LaRE/anns_dire/train_<gen>.txt
       LaRE/anns_dire/val_<gen>.txt
       (optional) LaRE/anns_dire/test_<gen>.txt
  2) Run extract_lare.py to produce:
       LaRE/features_<gen>_train/ann.txt
       LaRE/features_<gen>_val/ann.txt
       (optional) LaRE/features_<gen>_test/ann.txt

This script merges train/val (and test if present) ann.txt files into:
       LaRE/maps/map_<gen>.txt

Output format (matches original ann.txt):
   /abs/path/to/feature_xxx.pt<TAB>label

Usage in train_classifier_wmap.py:
   --train_file LaRE/anns_dire/train_<gen>.txt
   --val_file   LaRE/anns_dire/val_<gen>.txt
   --map_file   LaRE/maps/map_<gen>.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent


def merge_ann_files(src_files: List[Path], dst_file: Path) -> None:
    lines: List[str] = []
    for src in src_files:
        if not src.exists():
            print(f"[warn] ann.txt not found, skip: {src}")
            continue
        content = src.read_text(encoding="utf-8").splitlines()
        # Preserve original lines including tab separators
        for line in content:
            line = line.strip()
            if not line:
                continue
            lines.append(line + "\n")

    if not lines:
        print(f"[warn] no lines collected for {dst_file.name}, nothing written.")
        return

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.write_text("".join(lines), encoding="utf-8")
    print(f"[write] {dst_file}  (from {len(src_files)} splits, {len(lines)} entries)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge ann.txt files from features_<gen>_train/val(/test) into map_<gen>.txt."
    )
    ap.add_argument(
        "--gens",
        type=str,
        default="adm,biggan,flux,glide,sdv5,vqdm",
        help="Comma-separated generator names (corresponding to features_<gen>_* directories).",
    )
    ap.add_argument(
        "--features-root",
        type=str,
        default=str(ROOT),
        help="Root directory containing features_*_train/val dirs (default: LaRE root).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="maps",
        help="Output directory for map_<gen>.txt files (relative to LaRE root).",
    )
    args = ap.parse_args()

    gens = [g.strip() for g in args.gens.split(",") if g.strip()]
    features_root = Path(args.features_root)
    out_root = ROOT / args.out_dir

    for gen in gens:
        train_ann = features_root / f"features_{gen}_train" / "ann.txt"
        val_ann = features_root / f"features_{gen}_val" / "ann.txt"
        dst = out_root / f"map_{gen}.txt"
        test_ann = features_root / f"features_{gen}_test" / "ann.txt"
        print(f"[gen={gen}] train_ann={train_ann}, val_ann={val_ann}, test_ann={test_ann}")
        merge_ann_files([train_ann, val_ann, test_ann], dst)


if __name__ == "__main__":
    main()
