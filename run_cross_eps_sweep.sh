#!/bin/bash
# Cross-Generator Eps Sweep: 3 detectors × 16 gen pairs × 4 eps
# step=100 (strongest attack), 100 real + 100 fake per pair
#
# Usage:
#   nohup bash run_cross_eps_sweep.sh 0,1,2,3          > logs_cross_eps/main.log 2>&1 &
#   bash run_cross_eps_sweep.sh 0,1,2,3 dire
#   bash run_cross_eps_sweep.sh 0,1,2,3 lare
#   bash run_cross_eps_sweep.sh 0,1,2,3 ae
#   bash run_cross_eps_sweep.sh 0,1,2,3 dire,lare,ae
#
# Required environment variables:
#   DATA_ROOT          - path to dataset root (default: ./data)
#   DIFFUSION_CKPT     - path to 256x256_diffusion_uncond.pt
#   LARE_CKPT_ROOT     - path to LaRE checkpoint root (contains ExpLaRE2_*/Val_best.pth)
#   AE_MODEL_PATH      - path to SD 1.5 model folder (for AEROBLADE)
#   AE_THR_CSV         - path to aeroblade calibration distances CSV
#   CONDA_DIRE_ENV     - path to conda/venv activate for DIRE/LaRE env (Python 3.12)
#   CONDA_AE_ENV       - path to conda/venv activate for AEROBLADE env (Python 3.11)

set -euo pipefail

# ── GPUs & detectors ──────────────────────────────────────────
if [ $# -ge 1 ]; then IFS=',' read -ra GPUS <<< "$1"; else GPUS=(0 1 2 3); fi
if [ $# -ge 2 ]; then IFS=',' read -ra RUN_DETS <<< "$2"; else RUN_DETS=(dire lare ae); fi

GENS=(adm flux sdv5 vqdm)
PAIRS=()
for tg in "${GENS[@]}"; do
    for eg in "${GENS[@]}"; do
        PAIRS+=("${tg}:${eg}")
    done
done

EPS_VALS=(0.00392157 0.00784314 0.01568627 0.03137255)
EPS_NAMES=("1_255"    "2_255"    "4_255"    "8_255")
EPS_DISPLAY=("1/255"   "2/255"    "4/255"    "8/255")

DIRE_STEPS=20
STEPS=100
PICK_REAL=104
PICK_FAKE=104

DATA_ROOT="${DATA_ROOT:-./data}"
LOG_ROOT="${LOG_ROOT:-./logs_cross_eps_sweep}"
mkdir -p "$LOG_ROOT"/{dire,lare,ae}

echo "============================================================"
echo "Cross-Generator Eps Sweep"
echo "============================================================"
echo "GPUs:      ${GPUS[*]}"
echo "Detectors: ${RUN_DETS[*]}"
echo "Pairs:     ${#PAIRS[@]} (4×4 generators)"
echo "Eps:       ${EPS_DISPLAY[*]}"
echo "Steps:     ${STEPS}"
echo "Samples:   ${PICK_REAL} real + ${PICK_FAKE} fake"
echo "Total tasks/det: $(( ${#PAIRS[@]} * ${#EPS_VALS[@]} ))"
echo "============================================================"

# ── Task list: pair:eps_idx ────────────────────────────────────
ALL_TASKS=()
for pair in "${PAIRS[@]}"; do
    for ei in "${!EPS_VALS[@]}"; do
        ALL_TASKS+=("${pair}:${ei}")
    done
done

# ── Generic dispatcher ─────────────────────────────────────────
dispatch_tasks() {
    local DET_NAME="$1" RUNNER_FN="$2"
    local NUM=${#ALL_TASKS[@]} NEXT=0
    echo ""
    echo "============================================================"
    echo "[${DET_NAME^^}] Starting ${NUM} tasks"
    echo "============================================================"

    PIDS=()
    for ((j=0; j<${#GPUS[@]}; j++)); do PIDS[$j]=0; done

    while true; do
        local all_done=true
        for ((j=0; j<${#GPUS[@]}; j++)); do
            GPU=${GPUS[$j]}
            if [ ${PIDS[$j]} -ne 0 ] && ps -p ${PIDS[$j]} > /dev/null 2>&1; then
                all_done=false; continue
            fi
            PIDS[$j]=0
            if [ $NEXT -lt $NUM ]; then
                TASK="${ALL_TASKS[$NEXT]}"
                TRAIN_GEN="${TASK%%:*}"; REST="${TASK#*:}"
                TEST_GEN="${REST%%:*}";  EPS_IDX="${REST#*:}"
                echo "  [GPU ${GPU}] ${TRAIN_GEN}_on_${TEST_GEN}  eps=${EPS_DISPLAY[$EPS_IDX]}"
                $RUNNER_FN "$TRAIN_GEN" "$TEST_GEN" "$EPS_IDX" "$GPU" &
                PIDS[$j]=$!
                NEXT=$((NEXT + 1))
                all_done=false
            fi
        done
        [ $NEXT -ge $NUM ] && $all_done && break
        sleep 3
    done
    echo "[${DET_NAME^^}] Done."
}

# ============================================================
# DIRE
# ============================================================
DIRE_DIR="${DIRE_DIR:-$(dirname "$0")/DiffPure}"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-/path/to/256x256_diffusion_uncond.pt}"
DIRE_EXP_ROOT="${DIRE_EXP_ROOT:-./DiffPure/exp_cross_eps_sweep}"
mkdir -p "$DIRE_EXP_ROOT"

run_dire() {
    local TRAIN_GEN="$1" TEST_GEN="$2" EPS_IDX="$3" GPU="$4"
    local EPS="${EPS_VALS[$EPS_IDX]}" EPS_TAG="${EPS_NAMES[$EPS_IDX]}"
    local TAG="${TRAIN_GEN}_on_${TEST_GEN}_eps${EPS_TAG}"

    local CLS_CKPT="${DIRE_CKPT_ROOT:-/path/to/dire/ckpts}/cls_${TRAIN_GEN}_t20/ckpt/model_epoch_best.pth"

    (
        [ -n "${CONDA_DIRE_ENV:-}" ] && source "$CONDA_DIRE_ENV"
        cd "$DIRE_DIR"
        CUDA_VISIBLE_DEVICES=${GPU} python -u attack_dire_e2e.py \
            --config "imagenet.yml" \
            --diffusion_model_path "$DIFFUSION_CKPT" \
            --classifier_name "dire_resnet50" \
            --classifier_ckpt "$CLS_CKPT" \
            --quiet_full False \
            --dire_image_size 256 \
            --dire_apply_imagenet_norm False \
            --dire_scale_to_m11 True \
            --domain imagenet \
            --data_root "$DATA_ROOT" \
            --real_subdir "real/test" \
            --fake_dirs "${TEST_GEN}/test" \
            --pick_real "$PICK_REAL" \
            --pick_fake "$PICK_FAKE" \
            --real_offset 2500 \
            --fake_offset 2500 \
            --t 20 \
            --ode_method "euler" \
            --ode_step_size 0.02 \
            --fix_rand True \
            --eot_reps 1 \
            --ode_dire_smooth_eps 5e-5 \
            --attack_version "custom" \
            --attack_type "apgd-ce" \
            --autoattack_n_iter "$DIRE_STEPS" \
            --attack_method "full" \
            --lp_norm "Linf" \
            --adv_eps "$EPS" \
            --adv_batch_size 1 \
            --stream_data True \
            --dataloader_workers 1 \
            --seed 42 \
            --data_seed 0 \
            --exp "$DIRE_EXP_ROOT" \
            --image_folder "$TAG" \
            --device "cuda" \
            --save_pt True
    ) > "${LOG_ROOT}/dire/${TAG}.log" 2>&1
}

# ============================================================
# LaRE
# ============================================================
LARE_SAVE_ROOT="${LARE_SAVE_ROOT:-./LaRE/adv_outputs/cross_eps_sweep_images}"
LARE_OUT_ROOT="${LARE_OUT_ROOT:-./LaRE/adv_outputs/cross_eps_sweep}"
mkdir -p "$LARE_SAVE_ROOT" "$LARE_OUT_ROOT"

run_lare() {
    local TRAIN_GEN="$1" TEST_GEN="$2" EPS_IDX="$3" GPU="$4"
    local EPS="${EPS_VALS[$EPS_IDX]}" EPS_TAG="${EPS_NAMES[$EPS_IDX]}"
    local TAG="${TRAIN_GEN}_on_${TEST_GEN}_eps${EPS_TAG}"

    (
        [ -n "${CONDA_DIRE_ENV:-}" ] && source "$CONDA_DIRE_ENV"
        CUDA_VISIBLE_DEVICES=${GPU} python "$(dirname "$0")/LaRE/cross_autoattack_lare_e2e.py" \
            --ann_root "$(dirname "$0")/LaRE/anns_dire" \
            --ckpt_root "${LARE_CKPT_ROOT:-/path/to/lare/checkpoints}" \
            --ckpt_pattern "ExpLaRE2_${TRAIN_GEN}_retrain_Log_v12052017/Val_best.pth" \
            --train_gen "${TRAIN_GEN}" --test_gen "${TEST_GEN}" \
            --model CLipClassifierWMapV6 --clip_type RN50 \
            --pretrained_model_name_or_path "${AE_MODEL_PATH:-/path/to/sd15}" \
            --sd_img_size 256 --t 200 --ensemble_size 4 \
            --dtype fp16 \
            --prompt "a photo" --use_prompt_template \
            --data_size 224 --batch_size 8 --aa_bs 8 \
            --eps "$EPS" \
            --apgd_iter "$STEPS" \
            --pick_real "$PICK_REAL" --pick_fake "$PICK_FAKE" \
            --save_dir "${LARE_SAVE_ROOT}/${TAG}" \
            --out_dir "${LARE_OUT_ROOT}/${TAG}"
    ) > "${LOG_ROOT}/lare/${TAG}.log" 2>&1
}

# ============================================================
# AeroBlade
# ============================================================
AE_SAVE_ROOT="${AE_SAVE_ROOT:-./aeroblade_output/cross_eps_sweep_images}"
AE_OUT_ROOT="${AE_OUT_ROOT:-./aeroblade_output/cross_eps_sweep}"
mkdir -p "$AE_SAVE_ROOT" "$AE_OUT_ROOT"

run_ae() {
    local TRAIN_GEN="$1" TEST_GEN="$2" EPS_IDX="$3" GPU="$4"
    local EPS="${EPS_VALS[$EPS_IDX]}" EPS_TAG="${EPS_NAMES[$EPS_IDX]}"
    local TAG="${TRAIN_GEN}_on_${TEST_GEN}_eps${EPS_TAG}"

    (
        [ -n "${CONDA_AE_ENV:-}" ] && source "$CONDA_AE_ENV"
        CUDA_VISIBLE_DEVICES=${GPU} \
        PYTHONPATH="$(dirname "$0")/aeroblade/src:${PYTHONPATH:-}" \
        python "$(dirname "$0")/aeroblade/cross_autoattack_aeroblade_e2e.py" \
            --data_root "$DATA_ROOT" \
            --repo_ids "${AE_MODEL_PATH:-/path/to/sd15}" \
            --distance_metric lpips_vgg_2 \
            --train_gen "${TRAIN_GEN}" --test_gen "${TEST_GEN}" \
            --data_size 512 --batch_size 1 --aa_bs 1 \
            --eps "$EPS" \
            --thr_csv "${AE_THR_CSV:-/path/to/distances_calib.csv}" \
            --distance_sign pos \
            --apgd_iter "$STEPS" \
            --pick_real "$PICK_REAL" --pick_fake "$PICK_FAKE" \
            --save_dir "${AE_SAVE_ROOT}/${TAG}" \
            --out_dir "${AE_OUT_ROOT}/${TAG}" \
            --device cuda
    ) > "${LOG_ROOT}/ae/${TAG}.log" 2>&1
}

# ============================================================
# Dispatch
# ============================================================
for det in "${RUN_DETS[@]}"; do
    case "$det" in
        dire) dispatch_tasks "dire" "run_dire" ;;
        lare) dispatch_tasks "lare" "run_lare" ;;
        ae)   dispatch_tasks "ae"   "run_ae"   ;;
        *) echo "[warn] unknown detector: $det" ;;
    esac
done

echo ""
echo "============================================================"
echo "All done. Logs: ${LOG_ROOT}/{dire,lare,ae}/"
echo "Results:"
echo "  DIRE:  ${DIRE_EXP_ROOT}/"
echo "  LaRE:  ${LARE_OUT_ROOT}/"
echo "  AE:    ${AE_OUT_ROOT}/"
echo "============================================================"
