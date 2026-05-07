#!/bin/bash

[ -n "${CONDA_DIRE_ENV:-}" ] && source "$CONDA_DIRE_ENV"


DIFFUSION_MODEL="${DIFFUSION_CKPT:-/path/to/DIRE/guided-diffusion/models/256x256_diffusion_uncond.pt}"
CONFIG="imagenet.yml"
DEVICE="cuda"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "$0")/purified_samples_cross_pgd/adm}"

# GPU
GPU=4

T_VALUES=(10 20 30 50 100)

DIFFUSION_METHODS=("sde")

BATCH_SIZE=8
MAX_FILES=500

if [ ! -f "$DIFFUSION_MODEL" ]; then
    echo "ERROR: diffusion model not found: $DIFFUSION_MODEL"
    exit 1
fi

DIRE_BASE="${DIRE_ADV_ROOT:-$(dirname "$0")/DiffPure/exp_attacks_dire_cross_pgd20_eps3137255_r2500_f2500}"

DIRS=(
    #"${DIRE_BASE}/real_cls_adm/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_adm_test"
    #"${DIRE_BASE}/fake_cls_adm_on_adm/dire_resnet50/ode_custom_apgd-ce/seed42/data0/adv_images_rerun/full_dire_resnet50_adm_test"
    "${PROJECT_ROOT:-$(dirname "$0")/aeroblade_output/cross_e2e_images_pgd100_re_fixed/adm}"
    "${PROJECT_ROOT:-$(dirname "$0")/aeroblade_output/cross_e2e_images_pgd100_re_fixed/adm_on_adm}"
    #"${PROJECT_ROOT:-$(dirname "$0")/LaRE/adv_outputs/cross_e2e_images_pgd100_re_fp16/adm}"
    #"${PROJECT_ROOT:-$(dirname "$0")/LaRE/adv_outputs/cross_e2e_images_pgd100_re_fp16/adm_on_adm}"
)

DIR_NAMES=(
    #"dire_cls_adm_real"
    #"dire_adm_on_adm"
    "ae_adm_real"
    "ae_adm_on_adm"
    #"lare_adm_real"
    #"lare_adm_on_adm"
)

echo "=========================================="
echo "DiffPure Purification (ADM) - Cross PGD"
echo "=========================================="
echo "Diffusion model: $DIFFUSION_MODEL"
echo "T values: ${T_VALUES[@]}"
echo "GPU: $GPU"
echo "Output root: $OUTPUT_ROOT"
echo "=========================================="

for t in "${T_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Current t: $t"
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

            recursive_flag=""
            if [[ "$dir" == *"aeroblade"* ]] || [[ "$dir" == *"LaRE"* ]]; then
                recursive_flag="--recursive"
            fi

            output_dir="${output_root_t}/${method}/${dir_name}"

            CUDA_VISIBLE_DEVICES=$GPU python3 "$(dirname "$0")/purify_adversarial_samples.py" \
                --config "$CONFIG" \
                --diffusion_model_path "$DIFFUSION_MODEL" \
                --input_dirs "$dir" \
                --output_root "$output_dir" \
                --t $t \
                --device "cuda" \
                --diffusion_type "$method" \
                --batch_size $BATCH_SIZE \
                --max_files $MAX_FILES \
                --no_pt \
                $recursive_flag

            if [ $? -ne 0 ]; then
                echo "    Error: processing failed"
            else
                echo "    Done: $dir_name"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "All ADM purification tasks finished."
echo "Results saved under: ${OUTPUT_ROOT}_t*/"
echo "=========================================="
