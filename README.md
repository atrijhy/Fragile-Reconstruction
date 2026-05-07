# Adversarial Attacks Against Deepfake Detectors

Code for the paper: **"Fragile-Reconstruction"** .

We study adversarial attacks against three state-of-the-art deepfake detectors —
[DIRE](https://github.com/ZhendongWang6/DIRE),
[LaRE²](https://github.com/zhipeng-wei/LaRE2), and
[AEROBLADE](https://github.com/jonasricker/aeroblade) — including:

- End-to-end AutoAttack (APGD-CE) under multiple epsilon budgets
- Cross-generator, cross-method, and cross-model transferability evaluation
- Adversarial purification defenses (DiffPure, FLUX.1-dev, SD1.5, VQ-Diffusion)
- End-to-end PGD adversarial training for DIRE and LaRE²

---

## Repository Layout

```
release_code/
├── run_cross_eps_sweep.sh              # Main launcher: attacks across eps / generators
│
├── DiffPure/
│   ├── attack_dire_e2e.py              # DIRE E2E AutoAttack script (APGD-CE)
│   ├── runners/                        # DiffPure SDE runner
│   ├── guided_diffusion/               # OpenAI guided diffusion model
│   ├── attacks/                        # Logged APGD, simple optimizers
│   ├── utils.py
│   └── configs/imagenet.yml
│
├── LaRE/
│   ├── cross_autoattack_lare_e2e.py    # LaRE² E2E AutoAttack script
│   ├── train_classifier_wmap_e2e_pgd.py# LaRE² PGD adversarial training
│   ├── model.py / model_base.py        # LaRE² classifier architecture
│   ├── extract_lare.py                 # Pre-compute LaRE loss maps
│   ├── make_dire_lare_lists.py         # Build train/val file lists
│   └── anns_dire/                      # Pre-built annotation files
│
├── aeroblade/
│   ├── cross_autoattack_aeroblade_e2e.py # AEROBLADE E2E AutoAttack script
│   ├── setup.py
│   └── src/aeroblade/                  # AEROBLADE library code
│
├── DIRE/
│   ├── train_classifier_e2e_pgd.py     # DIRE PGD adversarial training (DDP)
│   ├── train_classifier_e2e_pgd_dp.py  # DIRE PGD adversarial training (DataParallel)
│   ├── networks/                       # DIRE ResNet classifier
│   └── utils/                          # Trainer, dataset, eval utilities
│
├── shared_detectors.py                 # Unified DIREDetector / LaREDetector / AEDetector
│
├── eval_cross_model_transferability.py # Transferability across generators (4×4)
├── eval_cross_method_transferability.py# Transferability across detectors
├── eval_cross_both_transferability.py  # Cross-generator × cross-detector grid
│
├── purify_adversarial_samples.py       # DiffPure (guided diffusion) purification
├── purify_adversarial_samples_flux.py  # FLUX.1-dev purification
├── purify_adversarial_samples_sd15.py  # Stable Diffusion v1.5 purification
├── purify_adversarial_samples_vqdm.py  # VQ-Diffusion purification
│
├── run_purification_cross_adm.sh       # Purification launcher (ADM-generated samples)
├── run_purification_cross_flux.sh      # Purification launcher (FLUX-generated samples)
├── run_purification_cross_sd15.sh      # Purification launcher (SDv1.5-generated samples)
└── run_purification_cross_vqdm.sh      # Purification launcher (VQDM-generated samples)
```

---

## Requirements

Two separate conda environments are needed because AEROBLADE requires Python 3.11
while DIRE and LaRE² run on Python 3.12.

```bash
# Environment 1: DIRE + LaRE² (Python 3.12)
conda create -n dire_lare python=3.12
conda activate dire_lare
pip install -r requirements.txt

# Environment 2: AEROBLADE (Python 3.11)
conda create -n aeroblade_env python=3.11
conda activate aeroblade_env
pip install -r requirements.txt
cd aeroblade && pip install -e .
```

See `requirements.txt` for the full dependency list.

---

## Checkpoints and Model Paths

Set the following environment variables before running any script (or pass them as
`--arg` flags where supported — every script accepts argparse overrides):

| Variable | Description |
|---|---|
| `DIFFUSION_CKPT` | Path to `256x256_diffusion_uncond.pt` (OpenAI guided diffusion) |
| `DIFFPURE_CONFIG` | Path to `DiffPure/configs/imagenet.yml` (auto-detected if unset) |
| `PROJECT_ROOT` | Root of this repo (auto-detected from `__file__` if unset) |
| `DIRE_CKPT_ROOT` | Directory containing DIRE classifier checkpoints `cls_<gen>/` |
| `SD15_MODEL_PATH` | Path to Stable Diffusion v1.5 HuggingFace model directory |
| `SD21_MODEL_PATH` | Path to Stable Diffusion 2.1-base model directory (LaRE²) |
| `AE_CALIB_CSV` | Path to AEROBLADE calibration CSV (`distances_calib.csv`) |

The guided diffusion checkpoint (`256x256_diffusion_uncond.pt`) is available from the
[OpenAI guided-diffusion release](https://github.com/openai/guided-diffusion).

DIRE classifier checkpoints can be trained with `DIRE/train_classifier_e2e_pgd.py`
or obtained from the [DIRE project page](https://github.com/ZhendongWang6/DIRE).

---

## Data Layout

Scripts expect data organised as:

```
<data_root>/
├── real/
│   ├── train/   *.png
│   └── val/     *.png
├── adm/
│   ├── train/   *.png
│   └── val/     *.png
├── flux/
├── sdv5/
└── vqdm/
```

The four generators used in our experiments are **adm**, **flux**, **sdv5**, **vqdm**.

---

## Running Attacks

### Quick start (single epsilon, single generator)

```bash
export DIFFUSION_CKPT=/path/to/256x256_diffusion_uncond.pt
export DIRE_CKPT_ROOT=/path/to/dire/checkpoints
export DATA_ROOT=/path/to/data

python DiffPure/attack_dire_e2e.py \
    --gen adm \
    --eps 8 \
    --data_root $DATA_ROOT \
    --ckpt_root $DIRE_CKPT_ROOT \
    --diffusion_model_path $DIFFUSION_CKPT
```

### Full epsilon sweep (all generators, all detectors)

```bash
bash run_cross_eps_sweep.sh
```

Edit the variable block at the top of `run_cross_eps_sweep.sh` to set paths and
choose which detectors / generators / epsilon values to sweep.

---

## Evaluating Transferability

After generating adversarial examples, run one of the three evaluation scripts:

```bash
# Cross-generator (attack on one generator, test on others)
python eval_cross_model_transferability.py \
    --dire_adv_root /path/to/dire/adv \
    --lare_adv_root /path/to/lare/adv \
    --ae_adv_root   /path/to/aeroblade/adv \
    --output_dir    ./figures/cross_model

# Cross-method (attack on DIRE, test LaRE²/AEROBLADE, etc.)
python eval_cross_method_transferability.py \
    --dire_adv_root /path/to/dire/adv \
    --lare_adv_root /path/to/lare/adv \
    --ae_adv_root   /path/to/aeroblade/adv \
    --output_dir    ./figures/cross_method

# Combined cross-generator × cross-method grid
python eval_cross_both_transferability.py \
    --dire_adv_root /path/to/dire/adv \
    --lare_adv_root /path/to/lare/adv \
    --ae_adv_root   /path/to/aeroblade/adv \
    --output_dir    ./figures/cross_both
```

---

## Adversarial Purification

Run purification on pre-generated adversarial `.png` files:

```bash
# DiffPure (guided diffusion)
python purify_adversarial_samples.py \
    --input_dirs /path/to/adv/adm,/path/to/adv/flux \
    --output_root ./purified_diffpure \
    --diffusion_model_path $DIFFUSION_CKPT

# FLUX.1-dev
python purify_adversarial_samples_flux.py \
    --input_dirs /path/to/adv/adm \
    --output_root ./purified_flux

# SD v1.5
python purify_adversarial_samples_sd15.py \
    --input_dirs /path/to/adv/adm \
    --output_root ./purified_sd15 \
    --model_path $SD15_MODEL_PATH

# VQ-Diffusion
python purify_adversarial_samples_vqdm.py \
    --input_dirs /path/to/adv/adm \
    --output_root ./purified_vqdm
```

Or use the shell launchers for the full cross-generator sweep:

```bash
bash run_purification_cross_adm.sh
bash run_purification_cross_flux.sh
```

---

## Adversarial Training

Train a DIRE classifier with end-to-end PGD adversarial training:

```bash
# DDP (recommended for multi-GPU)
python DIRE/train_classifier_e2e_pgd.py \
    --gen adm \
    --data_root /path/to/data \
    --diffusion_model_path $DIFFUSION_CKPT \
    --gpus 0,1,2,3 \
    --e2e_eps 8 \
    --e2e_k 8

# DataParallel (simpler single-machine variant)
python DIRE/train_classifier_e2e_pgd_dp.py \
    --gen adm \
    --data_root /path/to/data \
    --diffusion_model_path $DIFFUSION_CKPT
```

Train a LaRE² classifier with E2E PGD:

```bash
python LaRE/train_classifier_wmap_e2e_pgd.py \
    --train_file LaRE/anns_dire/adm_train.txt \
    --val_file   LaRE/anns_dire/adm_val.txt \
    --sd_model_path $SD21_MODEL_PATH \
    --out_dir ./lare_adv_ckpt \
    --pgd_mode approx \
    --pgd_eps 0.03137 \
    --pgd_k 8
```

---

## PYTHONPATH

`shared_detectors.py` and the attack scripts import from the sub-packages at runtime.
Add the repo root to your Python path before running:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
# For AEROBLADE also install the package:
cd aeroblade && pip install -e . && cd ..
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  year    = {TODO},
}
```
