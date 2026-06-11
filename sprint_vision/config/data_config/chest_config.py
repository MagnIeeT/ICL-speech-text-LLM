from .master_config import DatasetConfig, DatasetName, render_instruction

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

# HVB-style class definitions (one concise clause per finding).
# DRAFT — sourced from standard chest-radiology descriptions; pending professor
# review before retraining (analogous to ICI sourcing HVB defs from SLUE).
# These feed BOTH the training instruction and the per-class mAP/AUC probe.
CHEST_DEFINITIONS = {
    "pleural_effusion":        "abnormal fluid accumulation in the pleural space between lung and chest wall",
    "nodule":                  "a small rounded opacity up to three centimetres within the lung",
    "pneumonia":               "lung parenchymal inflammation seen as airspace consolidation or infiltrate",
    "cardiomegaly":            "an enlarged cardiac silhouette with cardiothoracic ratio above one half",
    "hilar_enlargement":       "enlargement of the pulmonary hila from vessels or lymph nodes",
    "fracture_old":            "a healed rib or bony thoracic fracture with callus or deformity",
    "fibrosis":                "reticular scarring with volume loss from chronic lung injury",
    "aortic_calcification":    "calcium deposition within the wall of the aorta",
    "tortuous_aorta":          "an elongated, widened, or unfolded aortic contour",
    "thickened_pleura":        "abnormal thickening of the pleural lining along the lung surface",
    "TB":                      "tuberculosis-related change such as upper-lobe opacity, cavitation, or calcified nodes",
    "pneumothorax":            "air in the pleural space causing partial or complete lung collapse",
    "emphysema":               "lung hyperinflation with destroyed alveoli and lucent lung fields",
    "atelectasis":             "collapse or incomplete expansion of lung tissue with volume loss",
    "calcification":           "a focal calcium deposit within lung, lymph node, or pleura",
    "pulmonary_edema":         "fluid in the alveoli and interstitium, often with vascular congestion",
    "increased_lung_markings": "accentuated bronchovascular markings throughout the lungs",
    "elevated_diaphragm":      "abnormal elevation of one or both hemidiaphragms",
    "consolidation":           "alveolar filling that opacifies the lung and obscures the vessels",
}
assert set(CHEST_DEFINITIONS) == set(CHEST_DISEASE_ORDER)

CHEST_INTRO = (
    "Task: Medical Image Classification. Dataset: Chest X-Ray (ThoracicAbnormality). "
    "Identify all applicable findings from the following options:"
)
CHEST_TAIL = (
    "List all that apply separated by commas. If none are present, answer 'none'."
)

CHEST_CONFIG = DatasetConfig(
    name=DatasetName.CHEST,
    is_multi_label=True,
    label_names=CHEST_DISEASE_ORDER,
    instruction=render_instruction(
        CHEST_INTRO, CHEST_DISEASE_ORDER, CHEST_DEFINITIONS, CHEST_TAIL
    ),
    images_subdir="chest/images",
    train_file="trainval.txt",
    test_file="test_WithLabel.txt",
    class_definitions=CHEST_DEFINITIONS,
    instruction_intro=CHEST_INTRO,
)
