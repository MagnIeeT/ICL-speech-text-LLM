from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

MELD_EMOTION_CONFIG = DatasetConfig(
    name=DatasetType.MELD_EMOTION,
    paths={
        DatasetSplit.TRAIN: get_env_path("MELD_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("MELD_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("MELD_TEST_PATH"),
    },
    # Bullet legend in the house format ("- label: definition"), which is what the
    # paper's prompts use and what _strip_legend and the routed-legend generator in
    # multi_task_dataset.py both key on. The definitions are the previous
    # "Choose X if there is Y" guidelines reformatted, not rewritten, so the
    # semantics are unchanged.
    prompt_template="""You are an emotion recognition expert. Based on the input, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: No distinct emotional state
- joy: Happiness, excitement, delight, pleasure, or positive enthusiasm
- sadness: Unhappiness, sorrow, grief, disappointment, or regret
- anger: Irritation, rage, fury, annoyance, or hostility
- fear: Terror, anxiety, worry, concern, or nervousness
- disgust: Repulsion, distaste, revulsion, or strong dislike
- surprise: Astonishment, shock, amazement, or unexpected reaction

Guidelines:
- Choose exactly one emotion from the list above
- Choose neutral ONLY IF the input expresses no distinct emotional state""",
    valid_labels=["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("MELD_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("MELD_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("MELD_TEST_LOOKUP"),
    },
)
