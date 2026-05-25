#!/usr/bin/env bash
set -euo pipefail

# Run from /home/firo/Documents/workspace/OpenPCDet/tools inside the OpenPCDet container/environment.
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-80}
export PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}"

ROBUST_CFG=cfgs/custom_models/pointpillar_mixed_farm_road_rice_field_rice_robust.yaml
FARM_CFG=cfgs/custom_models/pointpillar_farm_road.yaml
RICE_CFG=cfgs/custom_models/pointpillar_rice_field.yaml

latest_ckpt() {
    local ckpt_dir=$1
    ls -t "${ckpt_dir}"/checkpoint_epoch_*.pth 2>/dev/null | head -n 1
}

python3 train.py --cfg_file "${ROBUST_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --epochs "${EPOCHS}" --extra_tag mixed_rice_robust
ROBUST_CKPT=$(latest_ckpt ../output/custom_models/pointpillar_mixed_farm_road_rice_field_rice_robust/mixed_rice_robust/ckpt)
python3 test.py --cfg_file "${FARM_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --ckpt "${ROBUST_CKPT}" --extra_tag robust_mixed_to_farm --eval_tag robust_mixed_to_farm
python3 test.py --cfg_file "${RICE_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --ckpt "${ROBUST_CKPT}" --extra_tag robust_mixed_to_rice --eval_tag robust_mixed_to_rice
python3 evaluate_agri_distance_bins.py
