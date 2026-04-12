from .master_config import DatasetType, DatasetSplit, DatasetConfig

VOXPOPULI_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI,
    paths={
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_train_5fewshots",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_validation_5fewshots",
        DatasetSplit.TEST: "/home/leapers/weights/harinis/ICL-speech-text-LLM/data/voxpopuli_test_50fewshots",
    },
    prompt_template="""You are an Entity Type Classification system. For the given input, identify which of the following entity types are present:

- law: Laws, regulations, directives, and legal frameworks
- norp: Nationalities, religious, or political groups
- org: Companies, agencies, institutions
- person: People, including fictional characters
- place: Countries, cities, locations
- quant: Numbers, quantities, percentages
- when: Dates, times, durations, periods

Guidelines:
1. Return ONLY the entity type if present (e.g., 'place', 'person')
2. Return 'none' if no entity types are found
3. Be precise in identifying entity types""",
    valid_labels=["law", "norp", "org", "person", "place", "quant", "when"],
    completion_key="normalized_combined_ner",
    text_key="normalized_text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_train_audio_lookup",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_validation_audio_lookup",
        DatasetSplit.TEST: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_test_audio_lookup",
    },
)
