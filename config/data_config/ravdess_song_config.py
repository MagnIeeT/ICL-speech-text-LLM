from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

RAVDESS_SONG_CONFIG = DatasetConfig(
    name=DatasetType.RAVDESS_SONG,
    paths={
        DatasetSplit.TRAIN: get_env_path("RAVDESS_SONG_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("RAVDESS_SONG_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("RAVDESS_SONG_TEST_PATH"),
    },
    prompt_template="""You are an emotion recognition expert. Based on the singing audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: No distinct emotional state
- calm: A relaxed, peaceful state
- happy: Happiness or positive excitement
- sad: Sorrow or disappointment
- angry: Irritation or rage
- fearful: Terror or anxiety

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the singing audio""",
    valid_labels=[
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
    ],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("RAVDESS_SONG_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("RAVDESS_SONG_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("RAVDESS_SONG_TEST_LOOKUP"),
    },
)