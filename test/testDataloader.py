import torch
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
import logging

# Load dataset config
cfg_file = '../tools/cfgs/custom_models/voxel_rcnn_car.yaml' 
cfg_from_yaml_file(cfg_file, cfg)

# Create logger
logger = logging.getLogger("OpenPCDet")
logger.setLevel(logging.INFO)

# Define class names (adjust based on your dataset)
class_names = ['Pedestrian']  # Modify as needed

# Create DataLoader
dataset, dataloader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=class_names,
    batch_size=1,
    dist=False,
    workers=4,
    logger=logger
)

for batch in dataloader:
    print("Batch keys:", batch.keys())  # Check available keys
    # Check the input shapes for each part of the batch
    print("points Input shape:", batch['points'].shape)  # Points tensor shape
    print("voxels Input shape:", batch['voxels'].shape)  # Voxelized point data
    print("gt_boxes Input shape:", batch['gt_boxes'].shape)  # Ground truth boxes shape
    print("voxel_num_points Input shape:", batch['voxel_num_points'].shape)  # Voxel num points shape

    break  # Only process the first batch

# output 2025.3.7
# inside method : Input Shape:  (9332, 4)
# inside method : Input Shape:  (12081, 4)
# inside method : Input Shape:  (12509, 4)
# inside method : Input Shape:  (11904, 4)
# inside method : Input Shape:  (15757, 4)
# inside method : Input Shape:  (9841, 4)
# inside method : Input Shape:  (12346, 4)
# Batch keys: dict_keys(['frame_id', 'points', 
# 'gt_boxes', 'flip_x', 'noise_rot', 'noise_scale', '
# lidar_aug_matrix', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'batch_size'])
# points Input shape: (15757, 5)
# gt_boxes Input shape: (1, 12, 8)
# inside method : Input Shape:  (9396, 4)

# Load model
# model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names), dataset=dataloader.dataset)
# model.eval()  # Set to evaluation mode

# Get one batch
# batch = next(iter(dataloader))

# Find the input tensor shape
# print("Batch keys:", batch.keys())  # Find relevant key (usually 'points')
# print("Input shape:", batch['points'].shape)  # Check the actual shape
