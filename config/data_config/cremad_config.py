from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

CREMAD_CONFIG = DatasetConfig(
    name=DatasetType.CREMAD,
    paths={
        DatasetSplit.TRAIN: get_env_path("CREMAD_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("CREMAD_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("CREMAD_TEST_PATH"),
    },
    prompt_template="""You are an emotion recognition expert. Based on the audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: No distinct emotional state
- happy: Happiness or positive excitement
- angry: Irritation or rage
- sad: Sorrow or disappointment
- fear: Terror or anxiety
- disgust: Repulsion or distaste

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the audio""",
    valid_labels=[
        "neutral",
        "happy",
        "angry",
        "sad",
        "fear",
        "disgust",
    ],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("CREMAD_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("CREMAD_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("CREMAD_TEST_LOOKUP"),
    },
)