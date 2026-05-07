#!/bin/bash


set -euo pipefail

### ============ Configuration ============
MODEL_PATH="${MODEL_PATH:-/path/to/hf_models/microsoft_vq-diffusion-ithq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "$0")/purified_samples_cross_pgd/vqdm}"
PY_SCRIPT="$(dirname "$0")/purify_adversarial_samples_vqdm.py"

GPU="${1:-5}"
shift || true
if [ $# -gt 0 ]; then
    T_VALUES=("$@")
else
    T_VALUES=(1 2 3 5 10)
fi

GUIDANCE_SCALE=1.0
IMAGE_SIZE=256
BATCH_SIZE=8
MAX_FILES=500
### =======================================

[ -n "${CONDA_DIRE_ENV:-}" ] && source "$CONDA_DIRE_ENV"

echo "=========================================="
echo "VQ-Diffusion (ITHQ) Purification - Cross PGD"
echo "Model: ${MODEL_PATH}"
echo "GPU: ${GPU}"
echo "T values: ${T_VALUES[@]}"
echo "=========================================="

DIRE_BASE="${DIRE_ADV_ROOT:-$(dirname "$0")/DiffPure/exp_attacks_dire_cross_pgd20_eps3137255_r2500_f2500}"

DIRS=(
    "${DIRE_BASE}/real_cls_vqdm/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_vqdm_test"
    "${DIRE_BASE}/fake_cls_vqdm_on_vqdm/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_vqdm_test"
    "${PROJECT_ROOT:-$(dirname "$0")/aeroblade_output/cross_e2e_images_pgd100_re_fixed/vqdm}"
    "${PROJECT_ROOT:-$(dirname "$0")/aeroblade_output/cross_e2e_images_pgd100_re_fixed/vqdm_on_vqdm}"
    "${PROJECT_ROOT:-$(dirname "$0")/LaRE/adv_outputs/cross_e2e_images_pgd100_re_fp16/vqdm}"
    "${PROJECT_ROOT:-$(dirname "$0")/LaRE/adv_outputs/cross_e2e_images_pgd100_re_fp16/vqdm_on_vqdm}"
)

DIR_NAMES=(
    "dire_cls_vqdm_real"
    "dire_vqdm_on_vqdm"
    "ae_vqdm_real"
    "ae_vqdm_on_vqdm"
    "lare_vqdm_real"
    "lare_vqdm_on_vqdm"
)

export CUDA_VISIBLE_DEVICES=${GPU}

for t in "${T_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing t = ${t}"
    echo "=========================================="

    output_root_t="${OUTPUT_ROOT}_t${t}"

    for i in "${!DIRS[@]}"; do
        dir="${DIRS[$i]}"
        dir_name="${DIR_NAMES[$i]}"

        if [ ! -d "$dir" ]; then
            echo "  Skip (not found): $dir"
            continue
        fi

        echo "  Processing: ${dir_name}"

        RECURSIVE_FLAG=""
        if [[ "$dir" == *"aeroblade"* ]] || [[ "$dir" == *"LaRE"* ]]; then
            RECURSIVE_FLAG="--recursive"
        fi

        output_dir="${output_root_t}/vqdm/${dir_name}"

        python3 "${PY_SCRIPT}" \
            --model_path "${MODEL_PATH}" \
            --input_dirs "${dir}" \
            --output_root "${output_dir}" \
            --t ${t} \
            --device cuda \
            --batch_size ${BATCH_SIZE} \
            --image_size ${IMAGE_SIZE} \
            --max_files ${MAX_FILES} \
            --guidance_scale ${GUIDANCE_SCALE} \
            ${RECURSIVE_FLAG}

        rc=$?
        if [ $rc -ne 0 ]; then
            echo "    Error: processing failed (rc=$rc)"
        else
            echo "    Done: ${dir_name}"
        fi
    done

    echo "t = ${t} complete."
done

echo ""
echo "=========================================="
echo "All VQ-Diffusion purification tasks finished."
echo "Results saved under: ${OUTPUT_ROOT}_t*/"
echo "=========================================="
