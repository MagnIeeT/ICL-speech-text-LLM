from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


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

    # ── HVB-style class definitions (chest/endo multi-label only) ──────────────
    # When provided, `instruction` is rendered from these as an HVB-shaped
    # "- label: definition" block (see render_def_block / render_instruction).
    # The SAME block is reused verbatim by the per-class mAP/AUC probe so that
    # definitions are visible wherever the model is scored — mirroring HVB,
    # where one prompt_template feeds training, validation, and inference.
    # `instruction_intro` is the lead-in sentence before the bullet list.
    # Both are None for colon (binary), which keeps its hand-written instruction.
    class_definitions: Optional[Dict[str, str]] = None
    instruction_intro: Optional[str] = None


def render_def_block(intro: str, label_names: List[str], class_definitions: Dict[str, str]) -> str:
    """
    Render the HVB-style definition block:

        {intro}

        - label_0: definition_0
        - label_1: definition_1
        ...

    Mirrors ICI's prompt_template layout (hvb_config.py:11-37): an intro line
    followed by one "- label: definition" line per class, in canonical
    label_names order.  This is the SINGLE source of the definition block —
    both the training instruction (render_instruction) and the per-class
    mAP/AUC probe (dataload/medfmc_prompts.build_per_class_prompt) build from
    it, so they can never drift.
    """
    lines = [f"- {lbl}: {class_definitions[lbl]}" for lbl in label_names]
    return f"{intro}\n\n" + "\n".join(lines)


def render_instruction(
    intro: str,
    label_names: List[str],
    class_definitions: Dict[str, str],
    tail: str,
) -> str:
    """
    Full training instruction = HVB-style definition block + task tail.

        {intro}

        - label: definition
        ...

        {tail}

    `tail` is the multi-label answer instruction, e.g.
    "List all that apply separated by commas. If none are present, answer 'none'."
    """
    return render_def_block(intro, label_names, class_definitions) + f"\n\n{tail}"


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
