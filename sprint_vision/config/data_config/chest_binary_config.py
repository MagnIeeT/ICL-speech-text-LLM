from .master_config import DatasetConfig, DatasetName, render_instruction
from .chest_config import CHEST_DISEASE_ORDER

# ── Anatomical-site grouping over CHEST_DISEASE_ORDER's 19 findings ───────────
# Source: memory current_status.md, Session 2026-07-21 "ChestDR Pulmonary/
# Extrapulmonary" thread — grouping stress-tested against all 19 chest labels.
# Two labels are genuinely ambiguous (hilar_enlargement, calcification) and
# are defaulted to PULMONARY here (hilar = perihilar/pulmonary vasculature
# entry; parenchymal calcification is the most common site for that finding).
#
# Real cluster data (1161 images, trainval.txt + test_WithLabel.txt) showed
# 49.4% of images contain findings from BOTH groups simultaneously (57.5% of
# images with any finding at all) — this RULED OUT a single-label split
# (it would have required an arbitrary precedence rule discarding ground
# truth for the majority of positive images). chest_binary is therefore
# multi-label (2 independent flags), exactly like the original 19-label
# chest task, just with n_labels=2. See chest_binary_to_llava.py for the
# label derivation (a plain OR-reduction — no precedence rule needed).
PULMONARY_LABELS = frozenset({
    "nodule", "pneumonia", "hilar_enlargement", "fibrosis", "TB",
    "emphysema", "atelectasis", "calcification", "pulmonary_edema",
    "increased_lung_markings", "consolidation",
})
EXTRAPULMONARY_LABELS = frozenset({
    "pleural_effusion", "cardiomegaly", "fracture_old", "aortic_calcification",
    "tortuous_aorta", "thickened_pleura", "pneumothorax", "elevated_diaphragm",
})
assert PULMONARY_LABELS | EXTRAPULMONARY_LABELS == set(CHEST_DISEASE_ORDER)
assert not (PULMONARY_LABELS & EXTRAPULMONARY_LABELS)

CHEST_BINARY_LABEL_ORDER = ["pulmonary", "extrapulmonary"]

CHEST_BINARY_DEFINITIONS = {
    "pulmonary": (
        "the primary abnormality is within the lung parenchyma, airways, or "
        "pulmonary vasculature (e.g. nodule, pneumonia, consolidation, fibrosis)"
    ),
    "extrapulmonary": (
        "the primary abnormality is outside the lung parenchyma: pleura, "
        "heart, great vessels, diaphragm, or thoracic skeleton (e.g. pleural "
        "effusion, cardiomegaly, rib fracture)"
    ),
}
assert set(CHEST_BINARY_DEFINITIONS) == set(CHEST_BINARY_LABEL_ORDER)

CHEST_BINARY_INTRO = (
    "Task: Medical Image Classification. Dataset: Chest X-Ray (Anatomical Site "
    "of Abnormality). An image may show findings in one or both categories. "
    "Identify which of the following apply:"
)
CHEST_BINARY_TAIL = (
    "List all that apply separated by commas. If none are present, answer 'none'."
)

# NOTE: images_subdir/train_file/test_file mirror CHEST_CONFIG (same source
# images) but are NOT consumed by medfmc_to_llava.py's generic converter for
# this dataset — that converter is explicitly blocked for "chest_binary" (see
# medfmc_to_llava.py) because its generic multi-label parser would silently
# read only the first 2 of the 19 raw columns instead of grouping all 19. Use
# chest_binary_to_llava.py instead, which derives from an already-converted
# chest_*.json.
CHEST_BINARY_CONFIG = DatasetConfig(
    name=DatasetName.CHEST_BINARY,
    is_multi_label=True,
    label_names=CHEST_BINARY_LABEL_ORDER,
    instruction=render_instruction(
        CHEST_BINARY_INTRO, CHEST_BINARY_LABEL_ORDER, CHEST_BINARY_DEFINITIONS, CHEST_BINARY_TAIL
    ),
    images_subdir="chest/images",
    train_file="trainval.txt",
    test_file="test_WithLabel.txt",
    class_definitions=CHEST_BINARY_DEFINITIONS,
    instruction_intro=CHEST_BINARY_INTRO,
)
