#!/bin/bash

# FLUX.1-dev flow-matching latent space purification

set -euo pipefail

### ============ Configuration ============
MODEL_PATH="${MODEL_PATH:-/path/to/hf_models/black-forest-labs_FLUX.1-dev}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "$0")/purified_samples_cross_pgd/flux}"
PY_SCRIPT="$(dirname "$0")/purify_adversarial_samples_flux.py"

GPU="${1:-7}"
shift || true
if [ $# -gt 0 ]; then
    T_VALUES=("$@")
else
    T_VALUES=(10 20 30 50 100)
fi

GUIDANCE_SCALE=1.0
NUM_INFERENCE_STEPS=28
IMAGE_SIZE=512
BATCH_SIZE=1
MAX_FILES=500
### =======================================

[ -n "${CONDA_DIRE_ENV:-}" ] && source "$CONDA_DIRE_ENV"

echo "=========================================="
echo "FLUX.1-dev Purification - Cross PGD"
echo "Model: ${MODEL_PATH}"
echo "GPU: ${GPU}"
echo "T values: ${T_VALUES[@]}"
echo "=========================================="

DIRE_BASE="${DIRE_ADV_ROOT:-$(dirname "$0")/DiffPure/exp_attacks_dire_cross_pgd20_eps3137255_r2500_f2500}"

DIRS=(
    #"${DIRE_BASE}/real_cls_flux/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_flux_test"
    #"${DIRE_BASE}/fake_cls_flux_on_flux/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_flux_test"
    "${PROJECT_ROOT:-$(dirname "$0")/aeroblade_output/cross_e2e_images_pgd100_re_fixed/flux}"
    "${PROJECT_ROOT:-$(dirname "$0")/aeroblade_output/cross_e2e_images_pgd100_re_fixed/flux_on_flux}"
    #"${PROJECT_ROOT:-$(dirname "$0")/LaRE/adv_outputs/cross_e2e_images_pgd100_re_fp16/flux}"
    #"${PROJECT_ROOT:-$(dirname "$0")/LaRE/adv_outputs/cross_e2e_images_pgd100_re_fp16/flux_on_flux}"
)

DIR_NAMES=(
    #"dire_cls_flux_real"
    #"dire_flux_on_flux"
    "ae_flux_real"
    "ae_flux_on_flux"
    #"lare_flux_real"
    #"lare_flux_on_flux"
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

        output_dir="${output_root_t}/flux/${dir_name}"

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
            --num_inference_steps ${NUM_INFERENCE_STEPS} \
            --no_pt \
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
echo "All FLUX purification tasks finished."
echo "Results saved under: ${OUTPUT_ROOT}_t*/"
echo "=========================================="
