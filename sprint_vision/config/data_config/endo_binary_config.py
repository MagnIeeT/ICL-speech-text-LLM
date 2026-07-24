from .master_config import DatasetConfig, DatasetName, render_instruction
from .endo_config import ENDO_CATEGORY_ORDER

# ── Morphological grouping over ENDO_CATEGORY_ORDER's 4 findings ──────────────
# Source: memory current_status.md, Endo "Inflammatory vs Mass lesion" thread.
# Ulcer and erosion are a depth-graded spectrum of the SAME mucosal-break
# process (see ENDO_DEFINITIONS in endo_config.py — erosion is explicitly
# defined as the superficial version of the defect ulcer describes: a break
# in the mucosa from loss of epithelium, not a growth). Polyp and tumor are
# both protruding, space-occupying lesions (tissue proliferation rather than
# tissue loss).
#
# Named "Inflammatory" / "Mass lesion" rather than "Inflammatory / Neoplastic":
# polyps include non-neoplastic subtypes (hyperplastic, inflammatory
# pseudopolyp), and endoscopic appearance alone cannot establish neoplasia —
# that requires histology (the project's own `tumor` definition already
# hedges "suspicious for neoplasm", not "is a neoplasm"). "Mass lesion" is a
# morphological claim the image can support; "Neoplastic" would not be.
#
# IMPORTANT — unlike ChestDR, real cross-group co-occurrence has NOT yet been
# measured for endo as of this writing. See the endo co-occurrence analysis
# script (scratchpad, run on the cluster against real endo label files) for
# the real-data check that should inform whether this grouping's class
# balance is usable. is_multi_label=True is used here regardless (mirroring
# endo's existing multi-label architecture — ENDO_CONFIG is already
# is_multi_label=True for the 4-class task), so no precedence rule is needed
# even if cross-group co-occurrence turns out to be high, exactly as with
# chest_binary.
INFLAMMATORY_LABELS = frozenset({"ulcer", "erosion"})
MASS_LESION_LABELS  = frozenset({"polyp", "tumor"})
assert INFLAMMATORY_LABELS | MASS_LESION_LABELS == set(ENDO_CATEGORY_ORDER)
assert not (INFLAMMATORY_LABELS & MASS_LESION_LABELS)

ENDO_BINARY_LABEL_ORDER = ["inflammatory", "mass_lesion"]

ENDO_BINARY_DEFINITIONS = {
    "inflammatory": (
        "a mucosal break or defect from loss of epithelium -- an ulcer or "
        "erosion -- rather than a growth"
    ),
    "mass_lesion": (
        "a protruding, space-occupying lesion projecting into the lumen -- "
        "a polyp or a mass suspicious for neoplasm -- rather than a mucosal "
        "defect"
    ),
}
assert set(ENDO_BINARY_DEFINITIONS) == set(ENDO_BINARY_LABEL_ORDER)

ENDO_BINARY_INTRO = (
    "Task: Medical Image Classification. Dataset: Endoscopy (Lesion Morphology). "
    "An image may show findings in one or both categories. Identify which of "
    "the following apply:"
)
ENDO_BINARY_TAIL = (
    "List all that apply separated by commas. If none are present, answer 'none'."
)

# NOTE: images_subdir/train_file/test_file mirror ENDO_CONFIG (same source
# images) but are NOT consumed by medfmc_to_llava.py's generic converter for
# this dataset — that converter explicitly blocks "endo_binary" (see
# medfmc_to_llava.py) because its generic multi-label parser would silently
# read only the first 2 of the 4 raw columns instead of grouping all 4. Use
# endo_binary_to_llava.py instead, which derives from an already-converted
# endo_*.json.
ENDO_BINARY_CONFIG = DatasetConfig(
    name=DatasetName.ENDO_BINARY,
    is_multi_label=True,
    label_names=ENDO_BINARY_LABEL_ORDER,
    instruction=render_instruction(
        ENDO_BINARY_INTRO, ENDO_BINARY_LABEL_ORDER, ENDO_BINARY_DEFINITIONS, ENDO_BINARY_TAIL
    ),
    images_subdir="endo/images",
    train_file="trainval.txt",
    test_file="test_WithLabel.txt",
    class_definitions=ENDO_BINARY_DEFINITIONS,
    instruction_intro=ENDO_BINARY_INTRO,
)
