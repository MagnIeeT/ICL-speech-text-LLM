from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

RAVDESS_CONFIG = DatasetConfig(
    name=DatasetType.RAVDESS,
    paths={
        DatasetSplit.TRAIN: get_env_path("RAVDESS_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("RAVDESS_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("RAVDESS_TEST_PATH"),
    },
    prompt_template="""You are an emotion recognition expert. Based on the audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: No distinct emotional state
- calm: A relaxed, peaceful state
- happy: Happiness or positive excitement
- sad: Sorrow or disappointment
- angry: Irritation or rage
- fearful: Terror or anxiety
- disgust: Repulsion or distaste
- surprised: Astonishment or shock

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the audio""",
    valid_labels=[
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    ],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("RAVDESS_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("RAVDESS_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("RAVDESS_TEST_LOOKUP"),
    },
)