#!/usr/bin/env python3
"""
Compute embedding distances between class symbols for each epoch/mode/dataset.

For each symbol set, extracts the model's input embedding vector for the
FIRST subword token of each symbol, then computes pairwise cosine distances
between all class symbols. Outputs a table correlating min/avg distance with F1.

Usage:
    conda run -n qwen python analysis/embedding_distances.py

No GPU needed — only reads the embedding matrix (model.get_input_embeddings()).
Can run on login node.
"""

import re
import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2AudioForConditionalGeneration

SYMBOL_FILE = os.path.join(os.path.dirname(__file__), "041203_qwen_meld_emotion_dspo_symbols.txt")
MODEL_PATH  = "Qwen/Qwen2-Audio-7B-Instruct"

# ---------------------------------------------------------------------------
# Parse symbol file
# ---------------------------------------------------------------------------

def parse_symbol_file(path):
    """
    Returns list of dicts:
      {epoch, phase, mode, dataset, f1, symbols: {label: symbol}, first_tokens: {label: str}}
    """
    entries = []
    current_epoch = current_phase = current_mode = None

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        m = re.match(r"EPOCH (\d+)/11\s+\[(\S+)\]", line)
        if m:
            current_epoch = int(m.group(1))
            current_phase = m.group(2)
            continue

        m = re.match(r"SYMBOLS\+TOKENS \[(\w+)\]:", line)
        if m:
            current_mode = m.group(1).lower()
            continue

        m = re.match(r"\s+\[(\w+)\]\s+F1=([\d.]+)", line)
        if m:
            current_ds   = m.group(1)
            current_f1   = float(m.group(2))
            current_syms = {}
            current_ftoks = {}
            entries.append({
                "epoch": current_epoch, "phase": current_phase,
                "mode": current_mode,   "dataset": current_ds,
                "f1": current_f1,       "symbols": current_syms,
                "first_tokens": current_ftoks,
            })
            continue

        # symbol line: "    label  → sym  2tok  ['tok1', 'tok2']"
        m = re.match(r"\s{4}(\S+)\s+→\s+(\S+)\s+\d+tok\s+\[(.+)\]", line)
        if m and entries:
            label   = m.group(1)
            sym     = m.group(2)
            toks    = [t.strip().strip("'") for t in m.group(3).split(",")]
            entries[-1]["symbols"][label]      = sym
            entries[-1]["first_tokens"][label] = toks[0] if toks else sym

    return [e for e in entries if e["symbols"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading tokenizer and embedding matrix...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load ONLY the embedding layer — much faster than full model, CPU is fine
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    emb_matrix = model.get_input_embeddings().weight.detach()  # (vocab_size, hidden)
    print(f"Embedding matrix: {emb_matrix.shape}")
    del model  # free RAM

    entries = parse_symbol_file(SYMBOL_FILE)
    print(f"Parsed {len(entries)} symbol sets from {SYMBOL_FILE}\n")

    print(f"{'Epoch':>5} {'Phase':<14} {'Mode':<7} {'Dataset':<15} {'F1':>5}  "
          f"{'MinDist':>7} {'AvgDist':>7} {'MinPair':<30}")
    print("-" * 105)

    results = []
    for e in entries:
        labels      = list(e["first_tokens"].keys())
        first_toks  = list(e["first_tokens"].values())

        # Get token IDs and embedding vectors for each class's first subword
        ids  = [tok.convert_tokens_to_ids(t) for t in first_toks]
        vecs = torch.stack([emb_matrix[i] for i in ids])          # (n_classes, hidden)
        vecs = F.normalize(vecs, dim=-1)

        # Pairwise cosine distances (1 - cosine_sim)
        sims    = vecs @ vecs.T                                    # (n, n)
        dists   = 1.0 - sims
        n       = len(labels)
        pairs   = [(dists[i, j].item(), labels[i], labels[j])
                   for i in range(n) for j in range(i+1, n)]
        if not pairs:
            continue
        min_dist  = min(p[0] for p in pairs)
        avg_dist  = sum(p[0] for p in pairs) / len(pairs)
        min_pair  = min(pairs, key=lambda x: x[0])

        results.append({**e, "min_dist": min_dist, "avg_dist": avg_dist,
                        "min_pair": f"{min_pair[1]}∩{min_pair[2]}"})

        print(f"{e['epoch']:>5} {e['phase']:<14} {e['mode']:<7} {e['dataset']:<15} "
              f"{e['f1']:>5.3f}  {min_dist:>7.4f} {avg_dist:>7.4f} "
              f"{min_pair[1]}∩{min_pair[2]} ({min_pair[0]:.4f})")

    # Correlation summary per dataset
    print("\n\n=== CORRELATION: F1 vs min_dist (per dataset) ===")
    for ds in ["voxceleb", "hvb", "voxpopuli", "meld_emotion"]:
        for mode in ["fixed", "fresh"]:
            subset = [r for r in results if r["dataset"] == ds and r["mode"] == mode]
            if len(subset) < 3:
                continue
            f1s   = [r["f1"]       for r in subset]
            dists = [r["min_dist"] for r in subset]
            # Pearson correlation
            n  = len(f1s)
            mf = sum(f1s)/n;   md = sum(dists)/n
            num = sum((f-mf)*(d-md) for f,d in zip(f1s,dists))
            sf  = (sum((f-mf)**2 for f in f1s)/n)**0.5
            sd  = (sum((d-md)**2 for d in dists)/n)**0.5
            r   = num/(n*sf*sd) if sf*sd > 0 else 0.0
            print(f"  {ds:<15} [{mode}]  n={n}  pearson_r={r:+.3f}  "
                  f"(+r = higher dist → higher F1, supports H3)")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "embedding_distances.tsv")
    with open(out_path, "w") as f:
        f.write("epoch\tphase\tmode\tdataset\tf1\tmin_dist\tavg_dist\tmin_pair\n")
        for r in results:
            f.write(f"{r['epoch']}\t{r['phase']}\t{r['mode']}\t{r['dataset']}\t"
                    f"{r['f1']}\t{r['min_dist']:.6f}\t{r['avg_dist']:.6f}\t{r['min_pair']}\n")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
