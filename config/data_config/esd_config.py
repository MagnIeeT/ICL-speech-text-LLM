from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

ESD_CONFIG = DatasetConfig(
    name=DatasetType.ESD,
    paths={
        DatasetSplit.TRAIN: get_env_path("ESD_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("ESD_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("ESD_TEST_PATH"),
    },
    prompt_template="""You are an emotion recognition expert. Based on the audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: No distinct emotional state
- happy: Happiness or positive excitement
- angry: Irritation or rage
- sad: Sorrow or disappointment
- surprise: Astonishment or shock

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the audio""",
    valid_labels=[
        "neutral",
        "happy",
        "angry",
        "sad",
        "surprise",
    ],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("ESD_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("ESD_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("ESD_TEST_LOOKUP"),
    },
)