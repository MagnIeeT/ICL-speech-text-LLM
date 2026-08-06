from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# HeySQuAD (human-spoken SQuAD questions) — GENERATIVE QA testbed.
# The passage is provided as TEXT; the QUESTION is spoken (audio); the model must answer
# in 1-2 words. This extends the symbol/legend story to a GENERATIVE task and is a pure
# INSTRUCTION-FOLLOWING probe: the legend here is the answer-format directive, not a class map.
# task_type="qa" routes evaluation to exact-match / token-F1 / format-compliance (answer <= 2 words).
HEYSQUAD_CONFIG = DatasetConfig(
    name=DatasetType.HEYSQUAD,
    paths={
        DatasetSplit.TRAIN: get_env_path("HEYSQUAD_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("HEYSQUAD_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("HEYSQUAD_TEST_PATH"),
    },
    prompt_template="""You are given a reading passage, followed by a spoken question about it. Answer the spoken question using the passage.

Passage:
{text}

Instructions:
- Answer with the shortest possible span from the passage — 1 or 2 words only
- Do not write a full sentence, and do not add any explanation
- Output only the answer""",
    valid_labels=None,          # free-form answer, not a fixed label set
    completion_key="answer",
    text_key="context",         # the reading passage, injected via {text}
    task_type="qa",
    max_new_tokens=16,          # short answers, but allow a couple tokens of slack
    audio_lookup_paths=None,    # few-shot (if ever used) samples from train split at runtime
)
