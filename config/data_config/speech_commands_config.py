from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# Keyword-spotting CAPABILITY-RETENTION probe (core-10, closed-set, ORIGINAL/transcription only).
# No definition legend (words need none) and NOT used with symbol modes — it tests whether
# symbol training degraded the base model's word recognition, not symbol-following.
SPEECH_COMMANDS_CONFIG = DatasetConfig(
    name=DatasetType.SPEECH_COMMANDS,
    # Capability-retention probe — evaluated only (val/test); never trained on, so no TRAIN split.
    paths={
        DatasetSplit.VAL: get_env_path("SPEECH_COMMANDS_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("SPEECH_COMMANDS_TEST_PATH"),
    },
    prompt_template="""You will hear a single spoken word. Respond with EXACTLY ONE word from the list below — the word you hear:

Options: yes, no, up, down, left, right, on, off, stop, go

Guidelines:
- Choose exactly one word from the list above
- Respond with the word only""",
    valid_labels=[
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    ],
    completion_key="word_label",
    text_key="text",
    max_new_tokens=8,
)
