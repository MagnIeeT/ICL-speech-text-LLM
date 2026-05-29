from .master_config import DatasetConfig, DatasetName

COLON_CONFIG = DatasetConfig(
    name=DatasetName.COLON,
    is_multi_label=False,
    label_names=["0", "1"],
    instruction=(
        "Task: Medical Image Classification. Dataset: Colon Histopathology. "
        "Is there a tumor? Answer with the label only (0 for No, 1 for Yes)."
    ),
    images_subdir="colon/images",
    train_file="trainval.txt",
    test_file="test_WithLabel.txt",
)
