from .master_config import DatasetType, DatasetSplit, DatasetConfig

MELD_EMOTION_CONFIG = DatasetConfig(
    name=DatasetType.MELD_EMOTION,
    paths={
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/meld_train_5fewshots",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/meld_validation_5fewshots",
        DatasetSplit.TEST: "/home/leapers/weights/harinis/ICL-speech-text-LLM/data/meld_test_50fewshots",
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
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/meld_train_audio_lookup",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/meld_validation_audio_lookup",
        DatasetSplit.TEST: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/meld_test_audio_lookup",
    },
)
