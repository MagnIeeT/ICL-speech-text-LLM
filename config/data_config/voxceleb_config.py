from .master_config import DatasetType, DatasetSplit, DatasetConfig

VOXCELEB_CONFIG = DatasetConfig(
    name=DatasetType.VOXCELEB,
    paths={
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxceleb_train_5fewshots",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxceleb_validation_5fewshots",
        DatasetSplit.TEST: "/home/leapers/weights/harinis/ICL-speech-text-LLM/data/voxceleb_test_50fewshots",
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
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxceleb_train_audio_lookup",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxceleb_validation_audio_lookup",
        DatasetSplit.TEST: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxceleb_test_audio_lookup",
    },
)
