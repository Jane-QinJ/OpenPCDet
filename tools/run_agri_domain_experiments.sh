#!/usr/bin/env bash
set -euo pipefail

# Run from /home/firo/Documents/workspace/OpenPCDet/tools inside the OpenPCDet environment/container.
# Example:
#   cd /home/firo/Documents/workspace/OpenPCDet/tools
#   CUDA_VISIBLE_DEVICES=0 bash run_agri_domain_experiments.sh

BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-80}
export PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}"

FARM_CFG=cfgs/custom_models/pointpillar_farm_road.yaml
RICE_CFG=cfgs/custom_models/pointpillar_rice_field.yaml
MIXED_CFG=cfgs/custom_models/pointpillar_mixed_farm_road_rice_field.yaml

latest_ckpt() {
    local ckpt_dir=$1
    ls -t "${ckpt_dir}"/checkpoint_epoch_*.pth 2>/dev/null | head -n 1
}

# Prepare the mixed-domain dataset metadata. Single-domain farm/rice infos already exist.
python3 prepare_agri_domain_datasets.py --rebuild
python3 -m pcdet.datasets.custom.custom_dataset create_custom_infos cfgs/dataset_configs/custom_mixed_farm_road_rice_field_dataset.yaml

# A: train and evaluate farm road in-domain.
python3 train.py --cfg_file "${FARM_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --epochs "${EPOCHS}" --extra_tag farm_road_in_domain
FARM_CKPT=$(latest_ckpt ../output/custom_models/pointpillar_farm_road/farm_road_in_domain/ckpt)

# C: train and evaluate rice field in-domain.
python3 train.py --cfg_file "${RICE_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --epochs "${EPOCHS}" --extra_tag rice_field_in_domain
RICE_CKPT=$(latest_ckpt ../output/custom_models/pointpillar_rice_field/rice_field_in_domain/ckpt)

# B and D: cross-domain tests.
python3 test.py --cfg_file "${RICE_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --ckpt "${FARM_CKPT}" --extra_tag farm_to_rice --eval_tag farm_to_rice
python3 test.py --cfg_file "${FARM_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --ckpt "${RICE_CKPT}" --extra_tag rice_to_farm --eval_tag rice_to_farm

# E/F: mixed-domain training, then evaluate separately on farm and rice.
python3 train.py --cfg_file "${MIXED_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --epochs "${EPOCHS}" --extra_tag mixed_farm_rice
MIXED_CKPT=$(latest_ckpt ../output/custom_models/pointpillar_mixed_farm_road_rice_field/mixed_farm_rice/ckpt)
python3 test.py --cfg_file "${FARM_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --ckpt "${MIXED_CKPT}" --extra_tag mixed_to_farm --eval_tag mixed_to_farm
python3 test.py --cfg_file "${RICE_CFG}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" --ckpt "${MIXED_CKPT}" --extra_tag mixed_to_rice --eval_tag mixed_to_rice
