from .base_config import DatasetType, DatasetSplit, DatasetConfig
import random

MELD_EMOTION_CONFIG = DatasetConfig(
    name=DatasetType.MELD_EMOTION,
    paths={
        # DatasetSplit.TRAIN: "/home/sriramg/aneeraj/storage/data/Random/meld/MELD_Text_Audio_train_random_topk10",
        # DatasetSplit.VAL: "/home/sriramg/aneeraj/storage/data/Random/meld/MELD_Text_Audio_validation_random_topk10",
        # DatasetSplit.TEST: "/home/sriramg/aneeraj/storage/data/Random/meld/MELD_Text_Audio_test_random_topk10",
        DatasetSplit.TRAIN: "/home/sriramg/aneeraj/storage/data/Embedding/meld/MELD_Text_Audio_train_embedding_topk10",
        DatasetSplit.VAL: "/home/sriramg/aneeraj/storage/data/Embedding/meld/MELD_Text_Audio_validation_embedding_topk10",
        DatasetSplit.TEST: "/home/sriramg/aneeraj/storage/data/Embedding/meld/MELD_Text_Audio_test_embedding_topk10",
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
    valid_labels=["noemotion", "joy", "sadness", "anger", "fear", "disgust", "surprise"],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/home/sriramg/aneeraj/storage/data/Audio/meld/MELD_Text_Audio_train_audio_lookup",
        DatasetSplit.VAL: "/home/sriramg/aneeraj/storage/data/Audio/meld/MELD_Text_Audio_validation_audio_lookup",
        DatasetSplit.TEST: "/home/sriramg/aneeraj/storage/data/Audio/meld/MELD_Text_Audio_test_audio_lookup",
    }
)

# Greek version with arbitrary greek letter mappings
MELD_EMOTION_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.MELD_EMOTION_GREEK,
    paths=MELD_EMOTION_CONFIG.paths,
    prompt_template="""You are an emotion recognition expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, gamma, delta, epsilon, zeta, eta.

Guidelines:
- Choose alpha if there is no distinct emotional state (neutral)
- Choose beta if there is happiness, excitement, delight, pleasure, or positive enthusiasm
- Choose gamma if there is unhappiness, sorrow, grief, disappointment, or regret
- Choose delta if there is irritation, rage, fury, annoyance, or hostility
- Choose epsilon if there is terror, anxiety, worry, concern, or nervousness
- Choose zeta if there is repulsion, distaste, revulsion, or strong dislike
- Choose eta if there is astonishment, shock, amazement, or unexpected reaction""",
    valid_labels=["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"],
    label_mapping={
        "neutral": "alpha", 
        "joy": "beta", 
        "sadness": "gamma", 
        "anger": "delta", 
        "fear": "epsilon", 
        "disgust": "zeta", 
        "surprise": "eta"
    },
    audio_lookup_paths=MELD_EMOTION_CONFIG.audio_lookup_paths,
    text_key=MELD_EMOTION_CONFIG.text_key,
    completion_key=MELD_EMOTION_CONFIG.completion_key
)


# MELD_EMOTION_GREEK_CONFIG = DatasetConfig(
#     name=DatasetType.MELD_EMOTION_GREEK,
#     paths=MELD_EMOTION_CONFIG.paths,
#     prompt_template="""You are an emotion analysis expert. Based on the input, identify the most likely emotion from these options: wiprol, cynbax, durfez, kantol, puzvit, jogdem, or helfun.

# Guidelines:
# - Choose wiprol for expressions of joy, happiness, delight, or enthusiasm
# - Choose cynbax for expressions of sadness, disappointment, or grief
# - Choose durfez for expressions of anger, irritation, annoyance, or rage
# - Choose kantol for neutral statements with no clear emotion
# - Choose puzvit for expressions of surprise, shock, or astonishment
# - Choose jogdem for expressions of fear, worry, or anxiety
# - Choose helfun for expressions of disgust or repulsion

# Respond with only the emotion label.""",
#     valid_labels=["wiprol", "cynbax", "durfez", "kantol", "puzvit", "jogdem", "helfun"],
#     completion_key=MELD_EMOTION_CONFIG.completion_key,
#     text_key=MELD_EMOTION_CONFIG.text_key,
#     audio_lookup_paths=MELD_EMOTION_CONFIG.audio_lookup_paths,
#     label_mapping={
#         "joy": "wiprol",
#         "sadness": "cynbax", 
#         "anger": "durfez",
#         "neutral": "kantol",
#         "surprise": "puzvit",
#         "fear": "jogdem",
#         "disgust": "helfun"
#     }
# )

# Descriptions for each emotion category
MELD_EMOTION_DESCRIPTIONS = [
    "no distinct emotional state",
    "happiness, excitement, delight, pleasure, or positive enthusiasm",
    "unhappiness, sorrow, grief, disappointment, or regret",
    "irritation, rage, fury, annoyance, or hostility",
    "terror, anxiety, worry, concern, or nervousness",
    "repulsion, distaste, revulsion, or strong dislike",
    "astonishment, shock, amazement, or unexpected reaction"
]

# 10 different permutations of the emotion labels
MELD_EMOTION_PERMUTATIONS = [
    # Original order
    ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"],
    
    # Permutation 2: Group by valence (positive/negative/neutral)
    ["neutral", "joy", "surprise", "sadness", "anger", "fear", "disgust"],
    
    # Permutation 3: Group by intensity (low to high)
    ["neutral", "sadness", "joy", "disgust", "surprise", "fear", "anger"],
    
    # Permutation 4: Group by basic emotions first (Ekman's six basic emotions)
    ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"],
    
    # Permutation 5: Alphabetical order
    ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    
    # Permutation 6: Reverse original
    ["surprise", "disgust", "fear", "anger", "sadness", "joy", "neutral"],
    
    # Permutation 7: Group by social emotions vs. survival emotions
    ["joy", "sadness", "neutral", "surprise", "anger", "fear", "disgust"],
    
    # Permutation 8: Group by approach vs. avoidance emotions
    ["joy", "anger", "surprise", "sadness", "fear", "disgust", "neutral"],
    
    # Permutation 9: Group by common vs. uncommon in conversation
    ["neutral", "joy", "anger", "sadness", "surprise", "fear", "disgust"],
    
    # Permutation 10: Group by complexity (simple to complex)
    ["neutral", "joy", "anger", "fear", "disgust", "sadness", "surprise"]
]

# 10 corresponding Greek permutations
MELD_EMOTION_GREEK_PERMUTATIONS = [
    # Original order
    ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"],
    
    # Permutation 2: Group by valence (positive/negative/neutral)
    ["alpha", "beta", "eta", "gamma", "delta", "epsilon", "zeta"],
    
    # Permutation 3: Group by intensity (low to high)
    ["alpha", "gamma", "beta", "zeta", "eta", "epsilon", "delta"],
    
    # Permutation 4: Group by basic emotions first
    ["beta", "gamma", "delta", "epsilon", "zeta", "eta", "alpha"],
    
    # Permutation 5: Alphabetical order (matching the emotional labels)
    ["delta", "zeta", "epsilon", "beta", "alpha", "gamma", "eta"],
    
    # Permutation 6: Reverse original
    ["eta", "zeta", "epsilon", "delta", "gamma", "beta", "alpha"],
    
    # Permutation 7: Group by social emotions vs. survival emotions
    ["beta", "gamma", "alpha", "eta", "delta", "epsilon", "zeta"],
    
    # Permutation 8: Group by approach vs. avoidance emotions
    ["beta", "delta", "eta", "gamma", "epsilon", "zeta", "alpha"],
    
    # Permutation 9: Group by common vs. uncommon in conversation
    ["alpha", "beta", "delta", "gamma", "eta", "epsilon", "zeta"],
    
    # Permutation 10: Group by complexity (simple to complex)
    ["alpha", "beta", "delta", "epsilon", "zeta", "gamma", "eta"]
]

# Create swap configurations using the permutations
MELD_EMOTION_SWAP_CONFIGS = []
for perm in MELD_EMOTION_PERMUTATIONS:
# for perm in MELD_EMOTION_GREEK_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(MELD_EMOTION_CONFIG.valid_labels, perm)}
    descriptions = {label: desc for label, desc in zip(MELD_EMOTION_CONFIG.valid_labels, MELD_EMOTION_DESCRIPTIONS)}
    MELD_EMOTION_SWAP_CONFIGS.append(DatasetConfig(
        prompt_template=f"""You are an emotion recognition expert. Based on the input, respond with EXACTLY ONE WORD from these options: {', '.join(perm)}.

Guidelines:
{chr(10).join(f'- Choose {label} if there is {descriptions[orig]}' for label, orig in zip(perm, MELD_EMOTION_CONFIG.valid_labels))}""",
        label_mapping=mapping,
        valid_labels=perm,
        name=DatasetType.MELD_EMOTION_SWAP,
        paths=MELD_EMOTION_CONFIG.paths,
        audio_lookup_paths=MELD_EMOTION_CONFIG.audio_lookup_paths,
        text_key=MELD_EMOTION_CONFIG.text_key,
        completion_key=MELD_EMOTION_CONFIG.completion_key
    ))

def get_meld_emotion_swap_config(randomize: bool = False):
    if randomize:
        return random.choice(MELD_EMOTION_SWAP_CONFIGS)
    else:
        # Always return the second config when not randomizing
        return MELD_EMOTION_SWAP_CONFIGS[1]
