import torch
from pcdet.models import build_network
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
import logging


# Create a logger
logger = logging.getLogger("OpenPCDet")
logger.setLevel(logging.INFO)
# Load the model configuration file
model_cfg_path = "../tools/cfgs/custom_models/voxel_rcnn_car.yaml"  # Update this path
cfg_from_yaml_file(model_cfg_path, cfg)

# Create dataset and dataloader
dataset, dataloader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG, 
    class_names=cfg.CLASS_NAMES, 
    batch_size=1,  # Adjust batch size as needed
    dist=False, 
    workers=1, 
    training=False,
    logger=logger  # Pass the logger here!
)

# Define the number of classes
num_class = len(cfg.CLASS_NAMES)

# Build the model
model = build_network(model_cfg=cfg.MODEL, num_class=num_class, dataset=dataset)
checkpoint = torch.load("../output/custom_models/voxel_rcnn_car/kizu1401_epoch80_voxelrcnn/ckpt/checkpoint_epoch_80.pth")
model.load_state_dict(checkpoint['model_state'])

model.eval()  # Set to evaluation mode

# Assuming your model has two inputs for points and voxels
# Create a dictionary that matches the model's expected input format

# Create dummy input tensors
dummy_input_points = torch.randn(1, 9721, 5)  # Adjust for batch size of 1
dummy_input_voxels = torch.randn(1, 6663, 5, 4)  # Adjust for batch size of 1

# Pass the inputs as a tuple to the model
torch.onnx.export(
    model, 
    (dummy_input_points, dummy_input_voxels),  # Pass inputs as a tuple
    "model.onnx", 
    input_names=["points", "voxels"], 
    output_names=["output"],
    dynamic_axes={
        "points": {0: "batch_size", 1: "num_points"}, 
        "voxels": {0: "batch_size", 1: "num_voxels", 2: "num_points_per_voxel"}
    }
)

print("ONNX model exported successfully!")