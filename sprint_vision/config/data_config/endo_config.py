from .master_config import DatasetConfig, DatasetName, render_instruction

# Column order confirmed from:
#   /home/harinisrireddykandula/MedFM/medfmc/datasets/medical_datasets.py  class Endoscopy
ENDO_CATEGORY_ORDER = ["ulcer", "erosion", "polyp", "tumor"]
assert len(ENDO_CATEGORY_ORDER) == 4

# HVB-style class definitions (one concise clause per finding).
# DRAFT — sourced from standard endoscopy descriptions; pending professor review
# before retraining.  These feed BOTH the training instruction and the per-class
# mAP/AUC probe.
ENDO_DEFINITIONS = {
    "ulcer":   "a mucosal break with loss of epithelium and a depressed, often coated base",
    "erosion": "a superficial mucosal defect not penetrating the muscularis mucosae",
    "polyp":   "a protruding mucosal lesion projecting into the lumen",
    "tumor":   "an abnormal mass lesion suspicious for neoplasm",
}
assert set(ENDO_DEFINITIONS) == set(ENDO_CATEGORY_ORDER)

ENDO_INTRO = (
    "Task: Medical Image Classification. Dataset: Endoscopy (ColorectalLesion). "
    "Identify all applicable findings from the following options:"
)
ENDO_TAIL = (
    "List all that apply separated by commas. If none are present, answer 'none'."
)

ENDO_CONFIG = DatasetConfig(
    name=DatasetName.ENDO,
    is_multi_label=True,
    label_names=ENDO_CATEGORY_ORDER,
    instruction=render_instruction(
        ENDO_INTRO, ENDO_CATEGORY_ORDER, ENDO_DEFINITIONS, ENDO_TAIL
    ),
    images_subdir="endo/images",
    train_file="trainval.txt",
    test_file="test_WithLabel.txt",
    class_definitions=ENDO_DEFINITIONS,
    instruction_intro=ENDO_INTRO,
)
