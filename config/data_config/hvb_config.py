from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# Definition-sensitivity ablation: same task/labels, only the wording after each "- action:" changes.
_PARA = """You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- acknowledge: signals that information has been understood or received
- answer_agree: voices agreement
- answer_dis: voices disagreement
- answer_general: a general reply to a question
- apology: says sorry or expresses regret
- backchannel: short listening feedback such as "uh-huh" or "mm-hmm"
- disfluency: speech repairs, repetitions, or self-corrections
- other: an action that fits none of the other categories
- question_check: a question asked to confirm understanding
- question_general: a general question seeking information
- question_repeat: a request for something to be repeated
- self: speech directed at oneself
- statement_close: a closing or concluding remark
- statement_general: a general statement or piece of information
- statement_instruct: an instruction or direction
- statement_open: an opening remark or greeting
- statement_problem: a statement describing an issue or problem
- thanks: an expression of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions"""

_TERSE = """You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- acknowledge: shows understanding
- answer_agree: agreement
- answer_dis: disagreement
- answer_general: general answer
- apology: apology
- backchannel: brief feedback (uh-huh)
- disfluency: speech repair or repetition
- other: none of the categories
- question_check: verifying question
- question_general: general question
- question_repeat: request to repeat
- self: self-directed speech
- statement_close: closing statement
- statement_general: general statement
- statement_instruct: instruction
- statement_open: opening or greeting
- statement_problem: problem statement
- thanks: thanks

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions"""

_VERBOSE = """You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- acknowledge: the speaker shows that they have understood or received the information conveyed to them
- answer_agree: the speaker expresses agreement with what has been said
- answer_dis: the speaker expresses disagreement with what has been said
- answer_general: the speaker gives a general response or reply to a question
- apology: the speaker expresses regret, remorse, or says sorry
- backchannel: the speaker gives brief verbal or textual listening feedback, such as "uh-huh" or "mm-hmm", without taking the floor
- disfluency: the speaker produces speech repairs, repetitions, false starts, or corrections
- other: a dialogue action that does not fit any of the other listed categories
- question_check: the speaker asks a question in order to verify or confirm their understanding
- question_general: the speaker asks a general, information-seeking question
- question_repeat: the speaker requests that something be repeated
- self: the speaker produces speech directed at themselves rather than at the other party
- statement_close: the speaker makes a concluding or closing statement
- statement_general: the speaker makes a general statement or provides information
- statement_instruct: the speaker gives an instruction or direction
- statement_open: the speaker makes an opening statement or greeting
- statement_problem: the speaker describes an issue, difficulty, or problem
- thanks: the speaker expresses gratitude or thanks

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions"""

HVB_CONFIG = DatasetConfig(
    name=DatasetType.HVB,
    paths={
        DatasetSplit.TRAIN: get_env_path("HVB_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("HVB_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("HVB_TEST_PATH"),
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
    prompt_variants={"paraphrase": _PARA, "terse": _TERSE, "verbose": _VERBOSE},
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
    is_multi_label=True,
    max_new_tokens=20,
    audio_lookup_paths={
        DatasetSplit.TRAIN: get_env_path("HVB_TRAIN_LOOKUP"),
        DatasetSplit.VAL: get_env_path("HVB_VAL_LOOKUP"),
        DatasetSplit.TEST: get_env_path("HVB_TEST_LOOKUP"),
    },
)
