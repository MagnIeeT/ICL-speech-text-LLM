from .master_config import DatasetConfig, DatasetName

# Column order confirmed from:
#   /home/harinis/MedFM/medfmc/datasets/medical_datasets.py  class Chest19
CHEST_DISEASE_ORDER = [
    "pleural_effusion", "nodule", "pneumonia", "cardiomegaly",
    "hilar_enlargement", "fracture_old", "fibrosis",
    "aortic_calcification", "tortuous_aorta", "thickened_pleura", "TB",
    "pneumothorax", "emphysema", "atelectasis", "calcification",
    "pulmonary_edema", "increased_lung_markings", "elevated_diaphragm",
    "consolidation",
]
assert len(CHEST_DISEASE_ORDER) == 19

CHEST_CONFIG = DatasetConfig(
    name=DatasetName.CHEST,
    is_multi_label=True,
    label_names=CHEST_DISEASE_ORDER,
    instruction=(
        "Task: Medical Image Classification. Dataset: Chest X-Ray (ThoracicAbnormality). "
        "Which of the following findings are present: "
        + ", ".join(CHEST_DISEASE_ORDER) + "? "
        "List all that apply separated by commas. "
        "If none are present, answer 'none'."
    ),
    images_subdir="chest/images",
    train_file="trainval.txt",
    test_file="test_WithLabel.txt",
)
