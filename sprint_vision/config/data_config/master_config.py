from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class DatasetName(str, Enum):
    COLON = "colon"
    CHEST = "chest"
    ENDO  = "endo"


@dataclass
class DatasetConfig:
    name:          DatasetName
    is_multi_label: bool
    label_names:   List[str]   # canonical class names in column order
    instruction:   str         # prompt template shown to the model
    images_subdir: str         # relative to medfmc_root, e.g. "colon/images"
    train_file:    str         # default training label file, e.g. "trainval.txt"
    test_file:     str         # evaluation label file, e.g. "test_WithLabel.txt"


from .colon_config import COLON_CONFIG
from .chest_config import CHEST_CONFIG
from .endo_config  import ENDO_CONFIG

DATASET_CONFIGS: Dict[DatasetName, DatasetConfig] = {
    DatasetName.COLON: COLON_CONFIG,
    DatasetName.CHEST: CHEST_CONFIG,
    DatasetName.ENDO:  ENDO_CONFIG,
}


def get_dataset_config(name: str) -> DatasetConfig:
    try:
        key = DatasetName(name)
    except ValueError:
        valid = [e.value for e in DatasetName]
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {valid}")
    return DATASET_CONFIGS[key]


__all__ = [
    "DatasetName",
    "DatasetConfig",
    "DATASET_CONFIGS",
    "get_dataset_config",
]
