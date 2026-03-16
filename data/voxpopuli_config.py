from .base_config import DatasetType, DatasetSplit, DatasetConfig
import random

VOXPOPULI_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI,
    paths={
        # DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_1fewshots",
        # DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_test_20fewshots",
        # DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_test_20fewshots",
        # DatasetSplit.TRAIN: "/data1/harshanj/data/Embedding/voxpopuli/asapp-slue_voxpopuli_train_embedding_topk5",
        # DatasetSplit.VAL: "/data1/harshanj/data/Embedding/voxpopuli/asapp-slue_voxpopuli_validation_embedding_topk5",
        # DatasetSplit.TEST: "/data1/harshanj/data/Embedding/voxpopuli/asapp-slue_voxpopuli_test_embedding_topk5",

        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_train_5fewshots",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_validation_5fewshots",
        DatasetSplit.TEST:"/home/leapers/weights/harinis/ICL-speech-text-LLM/data/voxpopuli_test_50fewshots",
        # DatasetSplit.TRAIN: "/data1/harshanj/data/new/random/voxpopuli/asapp-slue_voxpopuli_train_random_topk10",
        # DatasetSplit.VAL: "/data1/harshanj/data/new/random/voxpopuli/asapp-slue_voxpopuli_validation_random_topk10",
        # DatasetSplit.TEST: "/data1/harshanj/data/new/random/voxpopuli/asapp-slue_voxpopuli_test_random_topk10",
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
        # DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_audio_lookup",
        # DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_audio_lookup",
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_train_audio_lookup",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_validation_audio_lookup",
        DatasetSplit.TEST: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue_voxpopuli_test_audio_lookup",
    }
    # audio_lookup_paths={
    #     DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_1fewshots",
    #     DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_1fewshots",
    #     DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_1fewshots",
    # }
)

VOXPOPULI_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI_GREEK,
    paths=VOXPOPULI_CONFIG.paths,
    prompt_template="""You are an Entity Type Classification system. For the given input, identify which of the following entity types are present:

- Zeta1: Laws, regulations, directives, and legal frameworks
- Zeta2: Nationalities, religious, or political groups
- Zeta3: Companies, agencies, institutions
- Zeta4: People, including fictional characters
- Zeta5: Countries, cities, locations
- Zeta6: Numbers, quantities, percentages
- Zeta7: Dates, times, durations, periods

Guidelines:
1. Return ONLY the entity type if present (e.g., 'Zeta5', 'Zeta4')
2. Return 'None' if no entity types are found
3. Be precise in identifying entity types""",
    valid_labels=["zeta1", "zeta2", "zeta3", "zeta4", "zeta5", "zeta6", "zeta7"],
    completion_key="normalized_combined_ner",
    text_key="normalized_text",
    audio_lookup_paths=VOXPOPULI_CONFIG.audio_lookup_paths,
    label_mapping={
        "law": "zeta1",
        "norp": "zeta2",
        "org": "zeta3",
        "person": "zeta4",
        "place": "zeta5",
        "quant": "zeta6",
        "when": "zeta7"
    }
)

# VOXPOPULI_GREEK_CONFIG = DatasetConfig(
#     name=DatasetType.VOXPOPULI_GREEK,
#     paths=VOXPOPULI_CONFIG.paths,
#     prompt_template="""You are an Entity Type Classification system. For the given input, identify which of the following entity types are present:

# - Jukbam: Laws, regulations, directives, and legal frameworks
# - Nexpil: Nationalities, religious, or political groups
# - Vorzat: Companies, agencies, institutions
# - Fimtug: People, including fictional characters
# - Qelzod: Countries, cities, locations
# - Wacryl: Numbers, quantities, percentages
# - Xespuj: Dates, times, durations, periods

# Guidelines:
# 1. Return ONLY the entity type if present (e.g., 'Qelzod', 'Fimtug')
# 2. Return 'None' if no entity types are found
# 3. Be precise in identifying entity types""",
#     valid_labels=["jukbam", "nexpil", "vorzat", "fimtug", "qelzod", "wacryl", "xespuj"],
#     completion_key="normalized_combined_ner",
#     text_key="normalized_text",
#     audio_lookup_paths=VOXPOPULI_CONFIG.audio_lookup_paths,
#     label_mapping={
#         "law": "jukbam",
#         "norp": "nexpil",
#         "org": "vorzat",
#         "person": "fimtug",
#         "place": "qelzod",
#         "quant": "wacryl",
#         "when": "xespuj"
#     }
# )


# Define descriptions for VoxPopuli entity types
VOXPOPULI_DESCRIPTIONS = [
    "Laws, regulations, directives, and legal frameworks",
    "Nationalities, religious, or political groups",
    "Companies, agencies, institutions",
    "People, including fictional characters",
    "Countries, cities, locations",
    "Numbers, quantities, percentages",
    "Dates, times, durations, periods"
]

# 10 different permutations of the VoxPopuli labels
VOXPOPULI_PERMUTATIONS = [
    # Original order
    ["LAW", "NORP", "ORG", "PERSON", "PLACE", "QUANT", "WHEN"],
    
    # Permutation 2: Rotate once
    ["NORP", "ORG", "PERSON", "PLACE", "QUANT", "WHEN", "LAW"],
    
    # Permutation 3: Rotate twice
    ["ORG", "PERSON", "PLACE", "QUANT", "WHEN", "LAW", "NORP"],
    
    # Permutation 4: Rotate thrice
    ["PERSON", "PLACE", "QUANT", "WHEN", "LAW", "NORP", "ORG"],
    
    # Permutation 5: Rotate by 4
    ["PLACE", "QUANT", "WHEN", "LAW", "NORP", "ORG", "PERSON"],
    
    # Permutation 6: Rotate by 5
    ["QUANT", "WHEN", "LAW", "NORP", "ORG", "PERSON", "PLACE"],
    
    # Permutation 7: Rotate by 6
    ["WHEN", "LAW", "NORP", "ORG", "PERSON", "PLACE", "QUANT"],
    
    # Permutation 8: Group by entities representing people/groups
    ["PERSON", "NORP", "ORG", "PLACE", "LAW", "QUANT", "WHEN"],
    
    # Permutation 9: Group by abstract concepts first
    ["LAW", "WHEN", "QUANT", "NORP", "ORG", "PERSON", "PLACE"],
    
    # Permutation 10: Reverse original
    ["WHEN", "QUANT", "PLACE", "PERSON", "ORG", "NORP", "LAW"]
]

# 10 corresponding Greek permutations
VOXPOPULI_GREEK_PERMUTATIONS = [
    # Original order (matching Greek labels to original order)
    ["zeta1", "zeta2", "zeta3", "zeta4", "zeta5", "zeta6", "zeta7"],
    
    # Permutation 2: Rotate once 
    ["zeta2", "zeta3", "zeta4", "zeta5", "zeta6", "zeta7", "zeta1"],
    
    # Permutation 3: Rotate twice
    ["zeta3", "zeta4", "zeta5", "zeta6", "zeta7", "zeta1", "zeta2"],
    
    # Permutation 4: Rotate thrice
    ["zeta4", "zeta5", "zeta6", "zeta7", "zeta1", "zeta2", "zeta3"],
    
    # Permutation 5: Rotate by 4
    ["zeta5", "zeta6", "zeta7", "zeta1", "zeta2", "zeta3", "zeta4"],
    
    # Permutation 6: Rotate by 5
    ["zeta6", "zeta7", "zeta1", "zeta2", "zeta3", "zeta4", "zeta5"],
    
    # Permutation 7: Rotate by 6
    ["zeta7", "zeta1", "zeta2", "zeta3", "zeta4", "zeta5", "zeta6"],
    
    # Permutation 8: Group by entities representing people/groups
    ["zeta4", "zeta2", "zeta3", "zeta5", "zeta1", "zeta6", "zeta7"],
    
    # Permutation 9: Group by abstract concepts first
    ["zeta1", "zeta7", "zeta6", "zeta2", "zeta3", "zeta4", "zeta5"],
    
    # Permutation 10: Reverse original
    ["zeta7", "zeta6", "zeta5", "zeta4", "zeta3", "zeta2", "zeta1"]
]

# Create swap configurations using the new permutations
VOXPOPULI_SWAP_CONFIGS = []
for perm in VOXPOPULI_GREEK_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(VOXPOPULI_CONFIG.valid_labels, perm)}
    descriptions = {label: desc for label, desc in zip(VOXPOPULI_CONFIG.valid_labels, VOXPOPULI_DESCRIPTIONS)}
    VOXPOPULI_SWAP_CONFIGS.append(DatasetConfig(
        prompt_template=f"""You are an Entity Type Classification system. For the given input, identify which of the following entity types are present:

{chr(10).join(f'- {label}: {descriptions[orig]}' for label, orig in zip(perm, VOXPOPULI_CONFIG.valid_labels))}

Guidelines:
1. Return ONLY the entity type if present (e.g., '{perm[4]}', '{perm[3]}')
2. Return 'None' if no entity types are found
3. Be precise in identifying entity types""",
        label_mapping=mapping,
        valid_labels=perm,
        name=DatasetType.VOXPOPULI_SWAP,
        paths=VOXPOPULI_CONFIG.paths,
        audio_lookup_paths=VOXPOPULI_CONFIG.audio_lookup_paths,
        text_key=VOXPOPULI_CONFIG.text_key,
        completion_key=VOXPOPULI_CONFIG.completion_key
    ))

def get_voxpopuli_swap_config(randomize: bool = False):
    if randomize:
        return random.choice(VOXPOPULI_SWAP_CONFIGS)
    else:
        # Always return the second config when not randomizing
        return VOXPOPULI_SWAP_CONFIGS[1]