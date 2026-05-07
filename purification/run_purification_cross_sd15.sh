#!/bin/bash

[ -n "${CONDA_DIRE_ENV:-}" ] && source "$CONDA_DIRE_ENV"


MODEL_PATH="${MODEL_PATH:-/path/to/hf_models/sd15}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "$0")/purified_samples_cross_pgd/sd15}"
PY_SCRIPT="$(dirname "$0")/purify_adversarial_samples_sd15_sde.py"

GPU=6
T_VALUES=(10 20 30 50 100)
DIFFUSION_METHODS=("sde")

USE_BM=1
SDE_STEPS=2
GUIDANCE_SCALE=1.0
PROMPT=""
MAX_FILES=500
BATCH_SIZE=8
### =======================================

echo "=========================================="
echo "SD v1.5 Latent Space Purification - Cross PGD"
echo "Model: $MODEL_PATH"
echo "GPU: $GPU"
echo "T list: ${T_VALUES[@]}"
echo "=========================================="

DIRE_BASE="${DIRE_ADV_ROOT:-$(dirname "$0")/DiffPure/exp_attacks_dire_cross_pgd20_eps3137255_r2500_f2500}"

DIRS=(
    #"${DIRE_BASE}/real_cls_sdv5/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_sdv5_test"
    #"${DIRE_BASE}/fake_cls_sdv5_on_sdv5/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_sdv5_test"
    "${PROJECT_ROOT:-$(dirname "$0")/aeroblade_output/cross_e2e_images_pgd100_re_fixed/sdv5}"
    "${PROJECT_ROOT:-$(dirname "$0")/aeroblade_output/cross_e2e_images_pgd100_re_fixed/sdv5_on_sdv5}"
    #"${PROJECT_ROOT:-$(dirname "$0")/LaRE/adv_outputs/cross_e2e_images_pgd100_re_fp16/sdv5}"
    #"${PROJECT_ROOT:-$(dirname "$0")/LaRE/adv_outputs/cross_e2e_images_pgd100_re_fp16/sdv5_on_sdv5}"
)

DIR_NAMES=(
    #"dire_cls_sdv5_real"
    #"dire_sdv5_on_sdv5"
    "ae_sdv5_real"
    "ae_sdv5_on_sdv5"
    #"lare_sdv5_real"
    #"lare_sdv5_on_sdv5"
)

export CUDA_VISIBLE_DEVICES=${GPU}
echo "Exported CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for t in "${T_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing t = $t"
    echo "=========================================="

    output_root_t="${OUTPUT_ROOT}_t${t}"

    for method in "${DIFFUSION_METHODS[@]}"; do
        echo "  Method: ${method^^} (t=$t)"

        for i in "${!DIRS[@]}"; do
            dir="${DIRS[$i]}"
            dir_name="${DIR_NAMES[$i]}"

            if [ ! -d "$dir" ]; then
                echo "  Skip (not found): $dir"
                continue
            fi

            echo "  Processing: $dir_name"

            RECURSIVE_FLAG=""
            if [[ "$dir" == *"aeroblade"* ]] || [[ "$dir" == *"LaRE"* ]]; then
                RECURSIVE_FLAG="--recursive"
            fi

            output_dir="${output_root_t}/${method}/${dir_name}"

            CMD=(python3 "$PY_SCRIPT" \
                --model_path "$MODEL_PATH" \
                --input_dirs "$dir" \
                --output_root "$output_dir" \
                --t "$t" \
                --device "cuda" \
                --diffusion_type "$method" \
                --batch_size "$BATCH_SIZE" \
                --max_files "$MAX_FILES" \
                --no_pt \
                $RECURSIVE_FLAG)

            if [[ "$method" == "sde" ]]; then
                if [[ "$USE_BM" -eq 1 ]]; then
                    CMD+=(--use_bm)
                fi
                CMD+=(--sde_steps "$SDE_STEPS")
            fi

            if [[ -n "$PROMPT" ]] && (( $(echo "$GUIDANCE_SCALE > 1.0" | bc -l) )); then
                CMD+=(--guidance_scale "$GUIDANCE_SCALE" --prompt "$PROMPT")
            fi

            echo "    CMD: ${CMD[@]}"
            "${CMD[@]}"
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "    Error: processing failed (rc=$rc)"
            else
                echo "    Done: $dir_name"
            fi
        done

        echo "  Method ${method^^} (t=$t) complete."
    done

    echo "t = $t complete."
done

echo ""
echo "=========================================="
echo "All SD v1.5 tasks finished."
echo "Results saved under: ${OUTPUT_ROOT}_t*/"
echo "=========================================="
