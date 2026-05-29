from .master_config import DatasetConfig, DatasetName

# Column order confirmed from:
#   /home/harinis/MedFM/medfmc/datasets/medical_datasets.py  class Endoscopy
ENDO_CATEGORY_ORDER = ["ulcer", "erosion", "polyp", "tumor"]
assert len(ENDO_CATEGORY_ORDER) == 4

ENDO_CONFIG = DatasetConfig(
    name=DatasetName.ENDO,
    is_multi_label=True,
    label_names=ENDO_CATEGORY_ORDER,
    instruction=(
        "Task: Medical Image Classification. Dataset: Endoscopy (ColorectalLesion). "
        "Which of the following findings are present: "
        + ", ".join(ENDO_CATEGORY_ORDER) + "? "
        "List all that apply separated by commas. "
        "If none are present, answer 'none'."
    ),
    images_subdir="endo/images",
    train_file="trainval.txt",
    test_file="test_WithLabel.txt",
)
