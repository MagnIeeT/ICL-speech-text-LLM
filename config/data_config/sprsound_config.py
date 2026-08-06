from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# SPRSound (respiratory sounds) — RECORD-level classification (whole recording → one label).
# OPAQUE-LABEL testbed: labels are clinical abbreviations (CAS/DAS) the base model has no prior
# for — the DESCRIPTION is what carries the meaning. This is where the definition/legend should
# win most over a bare label. "Poor Quality" recordings are dropped in processing.
SPRSOUND_CONFIG = DatasetConfig(
    name=DatasetType.SPRSOUND,
    paths={
        DatasetSplit.TRAIN: get_env_path("SPRSOUND_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("SPRSOUND_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("SPRSOUND_TEST_PATH"),
    },
    prompt_template="""You are a respiratory-sound analysis expert. The audio is a lung/breath sound recorded with a stethoscope. Respond with EXACTLY ONE of the following options:

Sound categories:
- normal: a clear breath sound with no added (adventitious) sounds
- cas: continuous adventitious sounds — musical sounds lasting longer than about 250 ms, such as wheezes (high-pitched whistling) or rhonchi (low-pitched snoring), caused by narrowed airways
- das: discontinuous adventitious sounds — brief, explosive crackling or popping sounds (crackles) caused by fluid or the sudden opening of small airways
- cas and das: both continuous sounds (wheeze/rhonchi) and discontinuous sounds (crackles) are present together

Guidelines:
- Choose exactly one category from the list above
- Base your answer only on the audio""",
    valid_labels=[
        "normal",
        "cas",
        "das",
        "cas and das",
    ],
    completion_key="sound_label",
    text_key="text",
    audio_lookup_paths=None,   # unused (few-shot samples from the train split at runtime)
)
