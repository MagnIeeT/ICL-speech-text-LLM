from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# SPRSound (respiratory sounds) — RECORD-level classification (whole recording → one label).
# OPAQUE-LABEL testbed: labels are clinical abbreviations (CAS/DAS) the base model has no prior
# for — the DESCRIPTION is what carries the meaning. This is where the definition/legend should
# win most over a bare label. "Poor Quality" recordings are dropped in processing.
# Definition-sensitivity ablation: same task/labels, only the wording after each "- category:" changes.
# NOTE: clinical content — review for medical accuracy before running.
_PARA = """You are a respiratory-sound analysis expert. The audio is a lung/breath sound recorded with a stethoscope. Respond with EXACTLY ONE of the following options:

Sound categories:
- normal: a clean breath sound with no added abnormal (adventitious) sounds
- cas: continuous adventitious sounds — musical tones lasting more than roughly 250 ms, such as wheezes (high-pitched whistling) or rhonchi (low-pitched snoring), produced by narrowed airways
- das: discontinuous adventitious sounds — short, explosive crackle or pop sounds (crackles) produced by fluid or the sudden reopening of small airways
- cas and das: both continuous sounds (wheezes/rhonchi) and discontinuous sounds (crackles) occur together

Guidelines:
- Choose exactly one category from the list above
- Base your answer only on the audio"""

_TERSE = """You are a respiratory-sound analysis expert. The audio is a lung/breath sound recorded with a stethoscope. Respond with EXACTLY ONE of the following options:

Sound categories:
- normal: clear breath sound, no adventitious sounds
- cas: continuous adventitious sounds (wheezes / rhonchi)
- das: discontinuous adventitious sounds (crackles)
- cas and das: both wheezes/rhonchi and crackles present

Guidelines:
- Choose exactly one category from the list above
- Base your answer only on the audio"""

_VERBOSE = """You are a respiratory-sound analysis expert. The audio is a lung/breath sound recorded with a stethoscope. Respond with EXACTLY ONE of the following options:

Sound categories:
- normal: a clear, healthy breath sound in which no added or abnormal (adventitious) sounds can be heard
- cas: continuous adventitious sounds — musical, tonal sounds that persist for longer than approximately 250 milliseconds, such as wheezes (high-pitched whistling sounds) or rhonchi (low-pitched snoring sounds), typically caused by narrowing of the airways
- das: discontinuous adventitious sounds — brief, explosive, non-musical crackling or popping sounds (known as crackles), typically caused by fluid in the airways or the sudden opening of collapsed small airways
- cas and das: a combination in which both continuous adventitious sounds (wheezes or rhonchi) and discontinuous adventitious sounds (crackles) are present together in the same recording

Guidelines:
- Choose exactly one category from the list above
- Base your answer only on the audio"""

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
    prompt_variants={"paraphrase": _PARA, "terse": _TERSE, "verbose": _VERBOSE},
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
