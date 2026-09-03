from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# Definition-sensitivity ablation: same task/labels, only the wording after each "- emotion:" changes.
_PARA = """You are an emotion recognition expert. Based on the audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: the speaker shows no clear emotion
- happy: the speaker sounds happy or positively excited
- angry: the speaker sounds irritated or enraged
- sad: the speaker sounds sorrowful or disappointed
- fear: the speaker sounds afraid or anxious
- disgust: the speaker sounds repulsed or distasteful

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the audio"""

_TERSE = """You are an emotion recognition expert. Based on the audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: no emotion
- happy: happiness
- angry: anger
- sad: sadness
- fear: fear
- disgust: disgust

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the audio"""

_VERBOSE = """You are an emotion recognition expert. Based on the audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: the speaker's voice conveys no distinct or pronounced emotional state, sounding calm and matter-of-fact
- happy: the speaker's voice conveys happiness, joy, or positive excitement
- angry: the speaker's voice conveys irritation, annoyance, hostility, or rage
- sad: the speaker's voice conveys sorrow, unhappiness, or disappointment
- fear: the speaker's voice conveys terror, fright, or anxiety
- disgust: the speaker's voice conveys repulsion, revulsion, or distaste

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the audio"""

CREMAD_CONFIG = DatasetConfig(
    name=DatasetType.CREMAD,
    paths={
        DatasetSplit.TRAIN: get_env_path("CREMAD_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("CREMAD_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("CREMAD_TEST_PATH"),
    },
    prompt_template="""You are an emotion recognition expert. Based on the audio, respond with EXACTLY ONE WORD from the following options:

Available emotions:
- neutral: No distinct emotional state
- happy: Happiness or positive excitement
- angry: Irritation or rage
- sad: Sorrow or disappointment
- fear: Terror or anxiety
- disgust: Repulsion or distaste

Guidelines:
- Choose exactly one emotion from the list above
- Be precise in identifying the emotion expressed in the audio""",
    prompt_variants={"paraphrase": _PARA, "terse": _TERSE, "verbose": _VERBOSE},
    valid_labels=[
        "neutral",
        "happy",
        "angry",
        "sad",
        "fear",
        "disgust",
    ],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("CREMAD_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("CREMAD_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("CREMAD_TEST_LOOKUP"),
    },
)