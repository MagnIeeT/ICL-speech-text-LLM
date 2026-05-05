from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

VOXCELEB_CONFIG = DatasetConfig(
    name=DatasetType.VOXCELEB,
    paths={
        DatasetSplit.TRAIN: get_env_path("VOXCELEB_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("VOXCELEB_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("VOXCELEB_TEST_PATH"),
    },
    prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: positive, negative, or neutral.

Guidelines:
- Choose positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose neutral ONLY IF the statement is purely factual with zero emotional content""",
    valid_labels=["positive", "negative", "neutral"],
    completion_key="sentiment",
    text_key="normalized_text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("VOXCELEB_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("VOXCELEB_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("VOXCELEB_TEST_LOOKUP"),
    },
)
