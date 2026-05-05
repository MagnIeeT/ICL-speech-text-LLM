from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

VOXPOPULI_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI,
    paths={
        DatasetSplit.TRAIN: get_env_path("VOXPOPULI_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("VOXPOPULI_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("VOXPOPULI_TEST_PATH"),
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
        DatasetSplit.TRAIN: get_env_path("VOXPOPULI_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("VOXPOPULI_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("VOXPOPULI_TEST_LOOKUP"),
    },
)
