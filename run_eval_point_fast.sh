#!/bin/bash
set -e

# 配置
PROJECT_ROOT="/home/haoyang22/project/VGGT-Long"
MODEL_NAME="VGGT-Long-PRED"
PRED_PLY_DIR="${PROJECT_ROOT}/precomputed_preds"
VOXEL_SIZE=0.01
# 自动创建带参数的文件夹，区分快速模式
OUTPUT_DIR="${PROJECT_ROOT}/eval_results/points/eval_results_fast_${VOXEL_SIZE}_$(date +%Y%m%d_%H%M%S)"
EVAL_SCRIPT_DIR="${PROJECT_ROOT}/eval/src/eval/mv_recon"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         VGGT-Long 3D Reconstruction Evaluation           ║"
echo "║  MODE: FAST (Downsampling enabled, voxel_size=${VOXEL_SIZE})      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  [1/4] Project Root:    ${PROJECT_ROOT}"
echo "  [2/4] Prediction PLYs: ${PRED_PLY_DIR}"
echo "  [3/4] Output Dir:      ${OUTPUT_DIR}"
echo "  [4/4] Eval Script:     ${EVAL_SCRIPT_DIR}/launch.py"
echo ""

echo "[Check] Checking files..."
if [ ! -d "${PRED_PLY_DIR}" ]; then
    echo "[ERROR] Prediction directory not found: ${PRED_PLY_DIR}"
    exit 1
fi
if [ ! -f "${EVAL_SCRIPT_DIR}/launch.py" ]; then
    echo "[ERROR] Launch script not found: ${EVAL_SCRIPT_DIR}/launch.py"
    exit 1
fi
echo "[Check] All files OK."
echo ""

echo "[Run] Starting evaluation (FAST mode, voxel_size=${VOXEL_SIZE})..."
cd "${EVAL_SCRIPT_DIR}"

accelerate launch --num_processes 1 --main_process_port 29602 launch.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --pred_ply_dir "${PRED_PLY_DIR}" \
    --size 518 \
    --voxel_size ${VOXEL_SIZE}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Evaluation finished successfully!                  ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Results are in: ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""