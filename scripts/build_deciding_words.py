"""Nearest-neighbour definitions and deciding words for trap training.

For every training dataset and every definition wording (the default template
plus each entry in `prompt_variants`), this script:

  1. parses the legend lines "- label: definition" for the dataset's valid labels,
  2. finds each definition's nearest other definition in the same legend, by
     cosine similarity of sentence embeddings (default `BAAI/llm-embedder`, CLS
     pooling, as in utils/generate_fewshots.py). Embeddings compare meaning, so
     short definitions with no shared words ("irritation or rage" vs "repulsion
     or distaste") still get a sensible neighbour. `--similarity tfidf` is a
     no-download fallback that only sees shared words.
  3. applies hand overrides, if given (JSON {dataset: {label: neighbour}},
     applied to every wording). Overrides are a safety valve after review and
     are recorded per entry, so they can be reported.
  4. lists the deciding words: content words of the definition that do not occur
     in its nearest neighbour's definition (stopwords removed).

Used by trap training for two things: the nearest-neighbour swap partner, and the
deciding-word removal in the conditional term. Output is meant to be reviewed by
hand before training.

Usage (CPU is enough):
    python scripts/build_deciding_words.py \
        --datasets hvb cremad minds14_en \
        --overrides config/data_config/deciding_words_overrides.json \
        --out config/data_config/deciding_words.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from config.data_config.master_config import DATASET_CONFIGS  # noqa: E402

_LEGEND_LINE = re.compile(r'(?m)^[ \t]*-[ \t]*([^:\n]+?)[ \t]*:[ \t]*(.+?)[ \t]*$')
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Function words only. scikit-learn's ENGLISH_STOP_WORDS also contains content
# words ("bill", "call", "fire", "system", "back", "move"), which silently
# dropped "bill" from pay_bill's deciding words.
STOPWORDS = frozenset("""
a an the and or but nor if then than so as of at by for from in into on onto to
with without about over under between through during before after above below
up down out off again further is am are was were be been being have has had do
does did doing can could will would shall should may might must i me my we our
you your he him his she her it its they them their this that these those there
here what which who whom whose when where why how all any both each few more
most other some such no not only own same too very just also any etc
while because until whether per via upon within whilst
""".split())


def parse_legend(template: str, valid_labels):
    """{label: definition} for legend lines whose name is a valid label."""
    valid = {l.lower(): l for l in valid_labels}
    out = {}
    for name, definition in _LEGEND_LINE.findall(template):
        key = name.strip().lower()
        if key in valid and valid[key] not in out:
            out[valid[key]] = definition.strip()
    return out


def content_words(text: str):
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def tfidf_similarity(defs):
    tfidf = TfidfVectorizer(stop_words=list(STOPWORDS), lowercase=True).fit_transform(defs)
    return cosine_similarity(tfidf)


class Embedder:
    """CLS-pooled sentence embeddings, same recipe as utils/generate_fewshots.py."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def similarity(self, defs):
        enc = self.tokenizer(defs, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            emb = self.model(**enc)[0][:, 0].float().cpu().numpy()
        return cosine_similarity(emb)


def analyse_legend(legend, sim_fn, overrides=None):
    """Nearest neighbour and deciding words for each label in one legend."""
    labels = sorted(legend)
    if len(labels) < 2:
        return {}
    defs = [legend[l] for l in labels]
    sim = sim_fn(defs)
    overrides = overrides or {}
    result = {}
    for i, label in enumerate(labels):
        sim[i, i] = -1.0
        source = "auto"
        if overrides.get(label) in legend and overrides[label] != label:
            j = labels.index(overrides[label])
            source = "override"
        else:
            if label in overrides:
                print(f"[warn] override {label} -> {overrides[label]!r} ignored (not in legend)")
            j = int(sim[i].argmax())
        neighbour = labels[j]
        neighbour_words = set(content_words(defs[j]))
        seen, deciding = set(), []
        for w in content_words(defs[i]):
            if w not in neighbour_words and w not in seen:
                seen.add(w)
                deciding.append(w)
        result[label] = {
            "definition": defs[i],
            "neighbour": neighbour,
            "neighbour_definition": defs[j],
            "similarity": round(float(sim[i, j]), 3),
            "neighbour_source": source,
            "deciding_words": deciding,
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["hvb", "cremad", "minds14_en"])
    ap.add_argument("--out", default="config/data_config/deciding_words.json")
    ap.add_argument("--similarity", choices=["embedding", "tfidf"], default="embedding")
    ap.add_argument("--embedder", default="BAAI/llm-embedder")
    ap.add_argument("--overrides", default=None,
                    help="optional JSON {dataset: {label: neighbour}}")
    args = ap.parse_args()

    if args.similarity == "embedding":
        sim_fn = Embedder(args.embedder).similarity
    else:
        sim_fn = tfidf_similarity
    all_overrides = {}
    if args.overrides:
        with open(args.overrides) as f:
            all_overrides = json.load(f)

    by_name = {dt.value: cfg for dt, cfg in DATASET_CONFIGS.items()}
    output = {}
    for ds in args.datasets:
        cfg = by_name.get(ds)
        if cfg is None:
            sys.exit(f"unknown dataset {ds!r}")
        templates = {"orig": cfg.prompt_template}
        templates.update(cfg.prompt_variants or {})
        output[ds] = {}
        for variant, template in templates.items():
            legend = parse_legend(template, cfg.valid_labels or [])
            missing = sorted(set(cfg.valid_labels or []) - set(legend))
            if missing:
                print(f"[warn] {ds}/{variant}: labels without a legend line: {missing}")
            output[ds][variant] = analyse_legend(legend, sim_fn, all_overrides.get(ds))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Review table: one line per label, default wording only.
    for ds, variants in output.items():
        print(f"\n=== {ds} (orig wording) ===")
        for label, r in variants.get("orig", {}).items():
            flag = "  <-- no deciding words" if not r["deciding_words"] else ""
            src = " (override)" if r["neighbour_source"] == "override" else ""
            print(f"{label:22s} nn={r['neighbour']:22s}{src} sim={r['similarity']:.2f} "
                  f"deciding={r['deciding_words']}{flag}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
