"""LLM-clustering-into-user-defined-clusters eval.
One clustering (3 clusters), THREE label styles to test the paper claim:
  meaningful  -> expect all methods tie (label carries the meaning)
  neutral/symbol -> definition is load-bearing -> dpi should win
Same audio/val set; differs only in which label column + prompt names are used.
Data built by utils/build_custom_taxonomy.py (cols cat_meaningful/cat_neutral/cat_symbol).
Run: VALIDATION_MODES=original NUM_EXAMPLES=0 (zero-shot, definitions in prompt).
"""
from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# canonical cluster definitions (identical across styles/datasets)
_DEFS = {
    "c1": "The customer wants to look up or review existing account information (balances, transactions, limits, address).",
    "c2": "The customer wants to take an action or set something up — move money, or activate/change a product.",
    "c3": "The customer is reporting a problem or protecting the account (app errors, card issues, fraud, freeze, lost).",
}
_STYLE_LABELS = {
    "meaningful": {"c1": "info_lookup", "c2": "account_action", "c3": "problem_report"},
    "neutral":    {"c1": "category_1",  "c2": "category_2",     "c3": "category_3"},
    "symbol":     {"c1": "vfeld",       "c2": "qomr",           "c3": "tirsk"},
}
_STYLE_KEY = {"meaningful": "cat_meaningful", "neutral": "cat_neutral", "symbol": "cat_symbol"}


def _prompt(style):
    lbl = _STYLE_LABELS[style]
    lines = "\n".join(f"- {lbl[c]}: {_DEFS[c]}" for c in ("c1", "c2", "c3"))
    return (
        "You are given a spoken customer request. Assign it to EXACTLY ONE of the "
        "following categories, based on the description of each:\n\n"
        f"{lines}\n\n"
        "Respond with the category label only."
    )


def _cfg(name, path_env, style):
    p = get_env_path(path_env)
    return DatasetConfig(
        name=name,
        paths={DatasetSplit.TRAIN: p, DatasetSplit.VAL: p, DatasetSplit.TEST: p},  # val-only eval
        prompt_template=_prompt(style),
        valid_labels=[_STYLE_LABELS[style][c] for c in ("c1", "c2", "c3")],
        completion_key=_STYLE_KEY[style],
        text_key="text",
        max_new_tokens=12,
    )


# (enum, path_env, style)
_SPECS = [
    ("MINDS14_FR_CAT_MEAN", "MINDS14_FR_CAT3_PATH", "meaningful"),
    ("MINDS14_FR_CAT_NEU",  "MINDS14_FR_CAT3_PATH", "neutral"),
    ("MINDS14_FR_CAT_SYM",  "MINDS14_FR_CAT3_PATH", "symbol"),
    ("MINDS14_KO_CAT_MEAN", "MINDS14_KO_CAT3_PATH", "meaningful"),
    ("MINDS14_KO_CAT_NEU",  "MINDS14_KO_CAT3_PATH", "neutral"),
    ("MINDS14_KO_CAT_SYM",  "MINDS14_KO_CAT3_PATH", "symbol"),
    ("SKIT_CAT_MEAN",       "SKIT_CAT3_PATH",       "meaningful"),
    ("SKIT_CAT_NEU",        "SKIT_CAT3_PATH",       "neutral"),
    ("SKIT_CAT_SYM",        "SKIT_CAT3_PATH",       "symbol"),
]

def build_configs():
    return {getattr(DatasetType, e): _cfg(getattr(DatasetType, e), pe, st) for e, pe, st in _SPECS}
