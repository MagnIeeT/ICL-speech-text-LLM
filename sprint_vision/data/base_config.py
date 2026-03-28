from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

class DatasetType(str, Enum):
    # Original datasets
    VOXCELEB = "voxceleb"
    HVB = "hvb"
    VOXPOPULI = "voxpopuli"
    
    # Symbol/Greek versions
    VOXCELEB_GREEK = "voxceleb_greek"
    HVB_GREEK = "hvb_greek"
    VOXPOPULI_GREEK = "voxpopuli_greek"
    
    # Swap/Permuted versions
    VOXCELEB_SWAP = "voxceleb_swap"
    HVB_SWAP = "hvb_swap"
    VOXPOPULI_SWAP = "voxpopuli_swap"
    
    # New dataset
    VOXPOPULI_NEL = "voxpopuli_nel"
    
    # Added for SQA
    SQA = "sqa"
    
    # Added for VP_NEL
    VP_NEL = "vp_nel"

    # Added for MELD
    MELD = "meld"
    MELD_GREEK = "meld_greek"
    MELD_EMOTION = "meld_emotion"
    MELD_EMOTION_GREEK = "meld_emotion_greek"

    MELD_EMOTION_SWAP="meld_emotion_swap"

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"

@dataclass
class DatasetConfig:
    name: DatasetType
    paths: Dict[DatasetSplit, str]
    prompt_template: str
    valid_labels: Optional[List[str]]
    completion_key: str
    text_key: str
    audio_lookup_paths: Dict[DatasetSplit, str] = None
    label_mapping: Dict[str, str] = None
    
    # New optional generic fields for flexible configurations
    additional_text_keys: Dict[str, str] = None  # For multiple text fields like {'question': 'normalized_question_text'}
    additional_audio_keys: Dict[str, str] = None  # For multiple audio fields
    additional_metadata_keys: Dict[str, str] = None  # For any other metadata fields
    output_format: str = None  # New field to specify expected output format

    def get_path(self, split: DatasetSplit) -> str:
        return self.paths[split]
    
    def get_audio_lookup_path(self, split: DatasetSplit) -> Optional[str]:
        if self.audio_lookup_paths:
            return self.audio_lookup_paths.get(split)
        return None

