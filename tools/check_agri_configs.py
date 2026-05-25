from pathlib import Path
import sys

ROOT = Path('/home/firo/Documents/workspace/OpenPCDet')
sys.path.insert(0, str(ROOT))


from easydict import EasyDict
import yaml

from pcdet.datasets.custom.custom_dataset import CustomDataset
from pcdet.utils import common_utils


CONFIGS = [
    'tools/cfgs/dataset_configs/custom_farm_road_dataset.yaml',
    'tools/cfgs/dataset_configs/custom_rice_field_dataset.yaml',
    'tools/cfgs/dataset_configs/custom_mixed_farm_road_rice_field_dataset.yaml',
]


def resolve_data_path(cfg):
    path = Path(cfg.DATA_PATH)
    if not path.is_absolute():
        path = (ROOT / 'tools' / path).resolve()
    return path


def main():
    logger = common_utils.create_logger()
    for cfg_name in CONFIGS:
        cfg = EasyDict(yaml.safe_load((ROOT / cfg_name).read_text()))
        data_path = resolve_data_path(cfg)
        train_set = CustomDataset(cfg, ['Pedestrian'], training=True, root_path=data_path, logger=logger)
        test_set = CustomDataset(cfg, ['Pedestrian'], training=False, root_path=data_path, logger=logger)
        print('%s train=%d test=%d path=%s' % (cfg_name, len(train_set), len(test_set), data_path))


if __name__ == '__main__':
    main()
