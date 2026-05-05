from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

MELD_EMOTION_CONFIG = DatasetConfig(
    name=DatasetType.MELD_EMOTION,
    paths={
        DatasetSplit.TRAIN: get_env_path("MELD_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("MELD_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("MELD_TEST_PATH"),
    },
    prompt_template="""You are an emotion recognition expert. Based on the input, respond with EXACTLY ONE WORD from these options: neutral, joy, sadness, anger, fear, disgust, or surprise.

Guidelines:
- Choose joy if there is happiness, excitement, delight, pleasure, or positive enthusiasm
- Choose sadness if there is unhappiness, sorrow, grief, disappointment, or regret
- Choose anger if there is irritation, rage, fury, annoyance, or hostility
- Choose fear if there is terror, anxiety, worry, concern, or nervousness
- Choose disgust if there is repulsion, distaste, revulsion, or strong dislike
- Choose surprise if there is astonishment, shock, amazement, or unexpected reaction
- Choose neutral ONLY IF the statement expresses no distinct emotional state""",
    valid_labels=["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("MELD_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("MELD_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("MELD_TEST_LOOKUP"),
    },
)
