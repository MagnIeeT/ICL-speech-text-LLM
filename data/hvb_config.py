from .base_config import DatasetType, DatasetSplit, DatasetConfig
import random

_fixed_hvb_config = None

HVB_CONFIG = DatasetConfig(
    name=DatasetType.HVB,
    paths={
        # DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_train_20fewshots",
        # DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_test_20fewshots",
        # DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_validation_5fewshots_new",
        # DatasetSplit.TRAIN: "/data1/harshanj/data/Embedding/hvb/asapp-slue-phase-2_hvb_train_embedding_topk5",
        # DatasetSplit.VAL: "/data1/harshanj/data/Embedding/hvb/asapp-slue-phase-2_hvb_validation_embedding_topk5",
        # DatasetSplit.TEST: "/data1/harshanj/data/Embedding/hvb/asapp-slue-phase-2_hvb_test_embedding_topk5"

        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_train_5fewshots",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_validation_5fewshots",
        DatasetSplit.TEST: "/home/leapers/weights/harinis/ICL-speech-text-LLM/data/hvb_test_50fewshots",      
        # DatasetSplit.TRAIN: "/data1/harshanj/data/new/random/hvb/asapp-slue-phase-2_hvb_train_random_topk10",
        # DatasetSplit.VAL: "/data1/harshanj/data/new/random/hvb/asapp-slue-phase-2_hvb_validation_random_topk10",
        # DatasetSplit.TEST: "/data1/harshanj/data/new/random/hvb/asapp-slue-phase-2_hvb_test_random_topk10"
    },
    prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- acknowledge: Shows understanding or receipt of information
- answer_agree: Expresses agreement
- answer_dis: Expresses disagreement
- answer_general: General response to a question
- apology: Expression of regret or sorry
- backchannel: Brief verbal/textual feedback (like "uh-huh", "mm-hmm")
- disfluency: Speech repairs, repetitions, or corrections
- other: Actions that don't fit other categories
- question_check: Questions to verify understanding
- question_general: General information-seeking questions
- question_repeat: Requests for repetition
- self: Self-directed speech
- statement_close: Concluding statements
- statement_general: General statements or information
- statement_instruct: Instructions or directions
- statement_open: Opening statements or greetings
- statement_problem: Statements describing issues or problems
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
    valid_labels=[
        "acknowledge", "answer_agree", "answer_dis", "answer_general",
        "apology", "backchannel", "disfluency", "other",
        "question_check", "question_general", "question_repeat",
        "self", "statement_close", "statement_general",
        "statement_instruct", "statement_open", "statement_problem",
        "thanks"
    ],
    completion_key="dialog_acts",
    text_key="text",
    audio_lookup_paths={
        # DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_train_audio_lookup",
        # DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_test_audio_lookup",
        # # DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_test_20fewshots",
        # DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_validation_audio_lookup_new",
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_train_audio_lookup",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_validation_audio_lookup",
        DatasetSplit.TEST: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_test_audio_lookup",
    }
)

HVB_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.HVB_GREEK,
    paths=HVB_CONFIG.paths,
    prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- foo: Shows understanding or receipt of information
- bar: Expresses agreement
- baz: Expresses disagreement
- qux: General response to a question
- quux: Expression of regret or sorry
- corge: Brief verbal/textual feedback
- grault: Speech repairs, repetitions, or corrections
- garply: Actions that don't fit other categories
- waldo: Questions to verify understanding
- fred: General information-seeking questions
- plugh: Requests for repetition
- xyzzy: Self-directed speech
- thud: Concluding statements
- wibble: General statements or information
- wobble: Instructions or directions
- wubble: Opening statements or greetings
- flob: Statements describing issues or problems
- zoop: Expressions of gratitude""",
    valid_labels=[
        "foo", "bar", "baz", "qux", "quux", 
        "corge", "grault", "garply", "waldo", "fred",
        "plugh", "xyzzy", "thud", "wibble", "wobble",
        "wubble", "flob", "zoop"
    ],
    completion_key=HVB_CONFIG.completion_key,
    text_key=HVB_CONFIG.text_key,
    audio_lookup_paths=HVB_CONFIG.audio_lookup_paths,
    label_mapping={
        "acknowledge": "foo",
        "answer_agree": "bar",
        "answer_dis": "baz",
        "answer_general": "qux",
        "apology": "quux",
        "backchannel": "corge",
        "disfluency": "grault",
        "other": "garply",
        "question_check": "waldo",
        "question_general": "fred",
        "question_repeat": "plugh",
        "self": "xyzzy",
        "statement_close": "thud",
        "statement_general": "wibble",
        "statement_instruct": "wobble",
        "statement_open": "wubble",
        "statement_problem": "flob",
        "thanks": "zoop"
    }
)



# Create new config with transformed labels
# HVB_GREEK_CONFIG = DatasetConfig(
#     name=DatasetType.HVB_GREEK,  # You'll need to add this to DatasetType enum
#     paths=HVB_CONFIG.paths,
#     prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

# Available dialogue actions:
# - foo: Shows understanding or receipt of information
# - Bar: Expresses agreement
# - baz: Expresses disagreement
# - QuX: General response to a question
# - QuLux: Expression of regret or sorry
# - Corges: Brief verbal/textual feedback
# - Graconsult: Speech repairs, repetitions, or corrections
# - GsharpLY: Actions that don't fit other categories
# - WalDo: Questions to verify understanding
# - FRED: General information-seeking questions
# - PlHugh: Requests for repetition
# - Xanalyzzi: Self-directed speech
# - ThUD: Concluding statements
# - WIBbles: General statements or information
# - stubbles: Instructions or directions
# - Wbles: Opening statements or greetings
# - FLOB: Statements describing issues or problems
# - ZoHop: Expressions of gratitude

# Guidelines:
# - Multiple actions can apply to a single statement
# - List all applicable actions separated by commas
# - Consider the banking context when analyzing
# - Be precise in identifying the dialogue actions""",
#     valid_labels= ["foo", "Bar", "baz", "QuX", "QuLux", "Corges", "Graconsult", "GsharpLY", 
#      "WalDo", "FRED", "PlHugh", "Xanalyzzi", "ThUD", "WIBbles", "stubbles", 
#      "Wbles", "FLOB", "ZoHop"],
#     completion_key=HVB_CONFIG.completion_key,
#     text_key=HVB_CONFIG.text_key,
#     audio_lookup_paths=HVB_CONFIG.audio_lookup_paths,
#     label_mapping={
#         # Map original HVB labels to transformed Greek labels
#         "acknowledge": "foo",
#         "answer_agree": "Bar",
#         "answer_dis": "baz",
#         "answer_general": "QuX",
#         "apology": "QuLux",
#         "backchannel": "Corges",
#         "disfluency": "Graconsult",
#         "other": "GsharpLY",
#         "question_check": "WalDo",
#         "question_general": "FRED",
#         "question_repeat": "PlHugh",
#         "self": "Xanalyzzi",
#         "statement_close": "ThUD",
#         "statement_general": "WIBbles",
#         "statement_instruct": "stubbles",
#         "statement_open": "Wbles",
#         "statement_problem": "FLOB",
#         "thanks": "ZoHop"
#     }
# )


# New random nonsensical labels for HVB
# HVB_GREEK_CONFIG = DatasetConfig(
#     name=DatasetType.HVB_GREEK,
#     paths=HVB_CONFIG.paths,
#     prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

# Available dialogue actions:
# - Zibfex: Shows understanding or receipt of information
# - Morvak: Expresses agreement
# - Penlut: Expresses disagreement
# - Yutrix: General response to a question
# - Quzdem: Expression of regret or sorry
# - Jafnol: Brief verbal/textual feedback
# - Wipcor: Speech repairs, repetitions, or corrections
# - Brezuv: Actions that don't fit other categories
# - Kolfim: Questions to verify understanding
# - Sepnid: General information-seeking questions
# - Hathog: Requests for repetition
# - Vilmep: Self-directed speech
# - Dronyx: Concluding statements
# - Goptaz: General statements or information
# - Cleybs: Instructions or directions
# - Ruxwel: Opening statements or greetings
# - Tamjid: Statements describing issues or problems
# - Loxfer: Expressions of gratitude""",
#     valid_labels=[
#         "zibfex", "morvak", "penlut", "yutrix", "quzdem", 
#         "jafnol", "wipcor", "brezuv", "kolfim", "sepnid",
#         "hathog", "vilmep", "dronyx", "goptaz", "cleybs",
#         "ruxwel", "tamjid", "loxfer"
#     ],
#     completion_key=HVB_CONFIG.completion_key,
#     text_key=HVB_CONFIG.text_key,
#     audio_lookup_paths=HVB_CONFIG.audio_lookup_paths,
#     label_mapping={
#         "acknowledge": "zibfex",
#         "answer_agree": "morvak",
#         "answer_dis": "penlut",
#         "answer_general": "yutrix",
#         "apology": "quzdem",
#         "backchannel": "jafnol",
#         "disfluency": "wipcor",
#         "other": "brezuv",
#         "question_check": "kolfim",
#         "question_general": "sepnid",
#         "question_repeat": "hathog",
#         "self": "vilmep",
#         "statement_close": "dronyx",
#         "statement_general": "goptaz",
#         "statement_instruct": "cleybs",
#         "statement_open": "ruxwel",
#         "statement_problem": "tamjid",
#         "thanks": "loxfer"
#     }
# )

# Include HVB_PERMUTATIONS and HVB_SWAP_CONFIGS from original file
HVB_PERMUTATIONS = [
    # Original order
    HVB_CONFIG.valid_labels,
    # Add other permutations as in original file
    ["question_check", "question_general", "question_repeat", "acknowledge", 
     "answer_agree", "answer_dis", "answer_general", "apology", "backchannel", 
     "disfluency", "other", "self", "statement_close", "statement_general",
     "statement_instruct", "statement_open", "statement_problem", "thanks"],
    
    # Permutation 3: Rotate statements to front
    ["statement_close", "statement_general", "statement_instruct", "statement_open", 
     "statement_problem", "acknowledge", "answer_agree", "answer_dis", "answer_general", 
     "apology", "backchannel", "disfluency", "other", "question_check", 
     "question_general", "question_repeat", "self", "thanks"],
    
    # Permutation 4: Rotate answers to front
    ["answer_agree", "answer_dis", "answer_general", "acknowledge", "apology", 
     "backchannel", "disfluency", "other", "question_check", "question_general",
     "question_repeat", "self", "statement_close", "statement_general",
     "statement_instruct", "statement_open", "statement_problem", "thanks"],
    
    # Permutation 5: Group similar actions
    ["acknowledge", "backchannel", "disfluency", "self", "answer_agree", 
     "answer_dis", "answer_general", "question_check", "question_general",
     "question_repeat", "statement_close", "statement_general", "statement_instruct", 
     "statement_open", "statement_problem", "apology", "thanks", "other"],
    
    # Permutation 6: Reverse original
    ["thanks", "statement_problem", "statement_open", "statement_instruct", 
     "statement_general", "statement_close", "self", "question_repeat",
     "question_general", "question_check", "other", "disfluency", "backchannel", 
     "apology", "answer_general", "answer_dis", "answer_agree", "acknowledge"],
    
    # Permutation 7: Group by conversation flow
    ["statement_open", "question_general", "answer_general", "question_check", 
     "answer_agree", "answer_dis", "acknowledge", "backchannel", "disfluency",
     "question_repeat", "statement_general", "statement_problem", "statement_instruct", 
     "apology", "self", "other", "statement_close", "thanks"],
    
    # Permutation 8: Group by response type
    ["question_general", "question_check", "question_repeat", "answer_general", 
     "answer_agree", "answer_dis", "statement_general", "statement_open",
     "statement_close", "statement_problem", "statement_instruct", "acknowledge", 
     "backchannel", "disfluency", "self", "apology", "thanks", "other"],
    
    # Permutation 9: Alternate question/answer/statement
    ["question_general", "answer_general", "statement_general", "question_check", 
     "answer_agree", "statement_open", "question_repeat", "answer_dis",
     "statement_close", "acknowledge", "backchannel", "statement_problem", 
     "disfluency", "self", "statement_instruct", "apology", "thanks", "other"],
    
    # Permutation 10: Group by formality
    ["statement_instruct", "statement_general", "question_general", "answer_general", 
     "statement_problem", "question_check", "answer_agree", "answer_dis",
     "statement_open", "statement_close", "acknowledge", "question_repeat", 
     "backchannel", "disfluency", "self", "apology", "thanks", "other"]
]

GREEK_PERMUTATIONS = [
    # Original order
    ["foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply", 
     "waldo", "fred", "plugh", "xyzzy", "thud", "wibble", "wobble", 
     "wubble", "flob", "zoop"],
    
    # Group by question-related terms
    ["waldo", "fred", "plugh", "foo", "bar", "baz", "qux", "quux", 
     "corge", "grault", "garply", "xyzzy", "thud", "wibble", "wobble", 
     "wubble", "flob", "zoop"],
    
    # Group by statement-related terms
    ["thud", "wibble", "wobble", "wubble", "flob", "foo", "bar", "baz", 
     "qux", "quux", "corge", "grault", "garply", "waldo", "fred", 
     "plugh", "xyzzy", "zoop"],
    
    # Group by answer-related terms
    ["bar", "baz", "qux", "foo", "quux", "corge", "grault", "garply", 
     "waldo", "fred", "plugh", "xyzzy", "thud", "wibble", "wobble", 
     "wubble", "flob", "zoop"],
    
    # Group similar concepts
    ["foo", "corge", "grault", "xyzzy", "bar", "baz", "qux", "waldo", 
     "fred", "plugh", "thud", "wibble", "wobble", "wubble", "flob", 
     "quux", "zoop", "garply"],
    
    # Reverse original
    ["zoop", "flob", "wubble", "wobble", "wibble", "thud", "xyzzy", 
     "plugh", "fred", "waldo", "garply", "grault", "corge", "quux", 
     "qux", "baz", "bar", "foo"],
    
    # Group by conversation flow
    ["wubble", "fred", "qux", "waldo", "bar", "baz", "foo", "corge", 
     "grault", "plugh", "wibble", "flob", "wobble", "quux", "xyzzy", 
     "garply", "thud", "zoop"],
    
    # Group by response type
    ["fred", "waldo", "plugh", "qux", "bar", "baz", "wibble", "wubble", 
     "thud", "flob", "wobble", "foo", "corge", "grault", "xyzzy", 
     "quux", "zoop", "garply"],
    
    # Alternating pattern
    ["fred", "qux", "wibble", "waldo", "bar", "wubble", "plugh", "baz", 
     "thud", "foo", "corge", "flob", "grault", "xyzzy", "wobble", 
     "quux", "zoop", "garply"],
    
    # Group by formality
    ["wobble", "wibble", "fred", "qux", "flob", "waldo", "bar", "baz", 
     "wubble", "thud", "foo", "plugh", "corge", "grault", "xyzzy", 
     "quux", "zoop", "garply"]
] 



HVB_DESCRIPTIONS = [
    "Shows understanding or receipt of information",
    "Expresses agreement",
    "Expresses disagreement",
    "General response to a question",
    "Expression of regret or sorry",
    "Brief verbal/textual feedback (like 'uh-huh', 'mm-hmm')",
    "Speech repairs, repetitions, or corrections",
    "Actions that don't fit other categories",
    "Questions to verify understanding",
    "General information-seeking questions",
    "Requests for repetition",
    "Self-directed speech",
    "Concluding statements",
    "General statements or information",
    "Instructions or directions",
    "Opening statements or greetings",
    "Statements describing issues or problems",
    "Expressions of gratitude"
]

HVB_SWAP_CONFIGS = []
for perm in HVB_PERMUTATIONS:
# for perm in GREEK_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(HVB_CONFIG.valid_labels, perm)}
    descriptions = {label: desc for label, desc in zip(HVB_CONFIG.valid_labels, HVB_DESCRIPTIONS)}
    HVB_SWAP_CONFIGS.append(DatasetConfig(
        prompt_template=f"""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
{chr(10).join(f'- {label}: {descriptions[orig]}' for label, orig in zip(perm, HVB_CONFIG.valid_labels))}

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping=mapping,
        valid_labels=perm,
        name=DatasetType.HVB_SWAP,
        paths=HVB_CONFIG.paths,
        audio_lookup_paths=HVB_CONFIG.audio_lookup_paths,
        text_key=HVB_CONFIG.text_key,
        completion_key=HVB_CONFIG.completion_key
    ))

def get_hvb_swap_config(randomize: bool = False):
    if randomize:
        return random.choice(HVB_SWAP_CONFIGS)
    else:
        # Always return the second config when not randomizing
        return HVB_SWAP_CONFIGS[1]

