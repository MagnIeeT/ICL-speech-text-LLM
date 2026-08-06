from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class DatasetType(str, Enum):
    VOXCELEB = "voxceleb"
    HVB = "hvb"
    VOXPOPULI = "voxpopuli"
    MELD_EMOTION = "meld_emotion"
    RAVDESS = "ravdess" 
    ESD = "esd"
    CREMAD = "cremad"
    RAVDESS_SONG = "ravdess_song"
    SKIT_S2I = "skit_s2i"
    SPEECH_COMMANDS = "speech_commands"
    MINDS14_EN = "minds14_en"
    MINDS14_FR = "minds14_fr"
    MINDS14_KO = "minds14_ko"
    SPRSOUND = "sprsound"
    HEYSQUAD = "heysquad"


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
    is_multi_label: bool = False
    audio_lookup_paths: Dict[DatasetSplit, str] = None
    label_mapping: Dict[str, str] = None
    max_new_tokens: int = 8
    task_type: str = "classification"   # "classification" | "qa" (free-form extractive answer → EM/token-F1)

    def get_path(self, split: DatasetSplit) -> str:
        return self.paths[split]

    def get_audio_lookup_path(self, split: DatasetSplit) -> Optional[str]:
        if not self.audio_lookup_paths:
            return None
        return self.audio_lookup_paths.get(split)


from .voxceleb_config import VOXCELEB_CONFIG
from .hvb_config import HVB_CONFIG
from .voxpopuli_config import VOXPOPULI_CONFIG
from .meld_emotion_config import MELD_EMOTION_CONFIG
from .ravdess_config import RAVDESS_CONFIG 
from .esd_config import ESD_CONFIG
from .cremad_config import CREMAD_CONFIG
from .ravdess_song_config import RAVDESS_SONG_CONFIG
from .skit_s2i_config import SKIT_S2I_CONFIG
from .speech_commands_config import SPEECH_COMMANDS_CONFIG
from .minds14_config import MINDS14_EN_CONFIG, MINDS14_FR_CONFIG, MINDS14_KO_CONFIG
from .sprsound_config import SPRSOUND_CONFIG
from .heysquad_config import HEYSQUAD_CONFIG

DATASET_CONFIGS: Dict[DatasetType, DatasetConfig] = {
    DatasetType.VOXCELEB: VOXCELEB_CONFIG,
    DatasetType.HVB: HVB_CONFIG,
    DatasetType.VOXPOPULI: VOXPOPULI_CONFIG,
    DatasetType.MELD_EMOTION: MELD_EMOTION_CONFIG,
    DatasetType.RAVDESS: RAVDESS_CONFIG,
    DatasetType.ESD: ESD_CONFIG,
    DatasetType.CREMAD: CREMAD_CONFIG,
    DatasetType.RAVDESS_SONG: RAVDESS_SONG_CONFIG,
    DatasetType.SKIT_S2I: SKIT_S2I_CONFIG,
    DatasetType.SPEECH_COMMANDS: SPEECH_COMMANDS_CONFIG,
    DatasetType.MINDS14_EN: MINDS14_EN_CONFIG,
    DatasetType.MINDS14_FR: MINDS14_FR_CONFIG,
    DatasetType.MINDS14_KO: MINDS14_KO_CONFIG,
    DatasetType.SPRSOUND: SPRSOUND_CONFIG,
    DatasetType.HEYSQUAD: HEYSQUAD_CONFIG,
}


def get_dataset_config(dataset_type: DatasetType) -> DatasetConfig:
    config = DATASET_CONFIGS.get(dataset_type)
    if config is None:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return config


__all__ = [
    "DatasetType",
    "DatasetSplit",
    "DatasetConfig",
    "get_dataset_config",
]
