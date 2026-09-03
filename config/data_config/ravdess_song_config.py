from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# Definition-sensitivity ablation: same task/labels, only the wording after each "- emotion:" changes.
_PARA = """You are an emotion recognition expert. Based on the singing audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: the singer shows no clear emotion
- calm: the singer sounds relaxed and peaceful
- happy: the singer sounds happy or positively excited
- sad: the singer sounds sorrowful or disappointed
- angry: the singer sounds irritated or enraged
- fearful: the singer sounds afraid or anxious

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the singing audio"""

_TERSE = """You are an emotion recognition expert. Based on the singing audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: no emotion
- calm: calm
- happy: happiness
- sad: sadness
- angry: anger
- fearful: fear

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the singing audio"""

_VERBOSE = """You are an emotion recognition expert. Based on the singing audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: the singing voice conveys no distinct or pronounced emotional state, sounding even and matter-of-fact
- calm: the singing voice conveys a relaxed, peaceful, and untroubled state
- happy: the singing voice conveys happiness, joy, or positive excitement
- sad: the singing voice conveys sorrow, unhappiness, or disappointment
- angry: the singing voice conveys irritation, annoyance, hostility, or rage
- fearful: the singing voice conveys terror, fright, or anxiety

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the singing audio"""

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
    prompt_variants={"paraphrase": _PARA, "terse": _TERSE, "verbose": _VERBOSE},
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