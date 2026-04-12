from .master_config import DatasetType, DatasetSplit, DatasetConfig

HVB_CONFIG = DatasetConfig(
    name=DatasetType.HVB,
    paths={
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_train_5fewshots",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_validation_5fewshots",
        DatasetSplit.TEST: "/home/leapers/weights/harinis/ICL-speech-text-LLM/data/hvb_test_50fewshots",
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
        "thanks",
    ],
    completion_key="dialog_acts",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_train_audio_lookup",
        DatasetSplit.VAL: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_validation_audio_lookup",
        DatasetSplit.TEST: "/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/asapp/slue-phase-2_hvb_test_audio_lookup",
    },
)
