#!/usr/bin/env python3
"""Process HeySQuAD (human) into our {audio, context, answer, ...} QA format.

GENERATIVE / instruction-following testbed:
  - context  : the reading passage (TEXT, embedded into the prompt via {text})
  - audio    : the human-SPOKEN question
  - answer   : the gold span, FILTERED to 1-2 words so EM / format-compliance are meaningful

Only the validation split is used for evaluation. task_type="qa" in the config routes
evaluation to exact-match / token-F1 / format-compliance.

Usage:
  python utils/process_heysquad.py \
      --raw_dir ~/data/raw/heysquad_validation_raw --split validation --out_base ~/data
Then set HEYSQUAD_VAL_PATH in .env to the saved dir.
"""
import argparse, os
from collections import Counter
from datasets import load_from_disk, Dataset, Audio

TARGET_SR = 16000
MAX_ANSWER_WORDS = 2


def _gold_texts(ans):
    """answers is a LIST of {'answer_start', 'text'} dicts (SQuAD multi-annotator).
    Return the unique non-empty gold answer strings."""
    out, seen = [], set()
    if isinstance(ans, list):
        for e in ans:
            t = e.get("text") if isinstance(e, dict) else str(e)
            t = str(t).strip() if t else ""
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return out


def build_rows(raw):
    """Keep only ANSWERABLE questions (is_impossible=False) whose SHORTEST gold answer is
    1..MAX_ANSWER_WORDS words — that shortest gold is the canonical brief target."""
    rows, dropped_impossible, dropped_long = [], 0, 0
    for item in raw:
        if item.get("is_impossible"):
            dropped_impossible += 1
            continue
        golds = _gold_texts(item.get("answers"))
        if not golds:
            dropped_impossible += 1
            continue
        shortest = min(golds, key=lambda t: len(t.split()))
        if not (1 <= len(shortest.split()) <= MAX_ANSWER_WORDS):
            dropped_long += 1
            continue
        rows.append({
            "id": str(item.get("id", len(rows))),
            "audio": item.get("audio"),                  # spoken question (HF audio dict)
            "context": str(item.get("context", "")),
            "question": str(item.get("question", "")),   # original text question (reference only)
            "answer": shortest,                          # canonical brief answer (target)
            "answers_all": golds,                        # all unique golds (for stricter EM/F1 later)
            "text": str(item.get("context", "")),        # alias; text_key="context" is what's used
            "few_shot_examples": [],
        })
    print(f"kept {len(rows)}  (dropped {dropped_impossible} impossible/no-answer, "
          f"{dropped_long} shortest-gold >{MAX_ANSWER_WORDS}-word)")
    print("target answer word-count dist:", dict(sorted(Counter(len(r["answer"].split()) for r in rows).items())))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True, help="load_from_disk dir of the raw HeySQuAD split")
    p.add_argument("--split", default="validation")
    p.add_argument("--out_base", default=os.environ.get("BASE_DATA_DIR", os.path.expanduser("~/data")))
    args = p.parse_args()

    raw = load_from_disk(args.raw_dir)
    print(f"raw {args.split}: {len(raw)} rows; columns: {raw.column_names}")
    rows = build_rows(raw)
    out = os.path.join(args.out_base, f"heysquad_{args.split}")
    ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=TARGET_SR))
    ds.save_to_disk(out)
    print(f"saved {args.split}: {len(rows)} → {out}")
    print("Done. Set in .env: HEYSQUAD_VAL_PATH → this dir (use split=validation at inference).")


if __name__ == "__main__":
    main()
