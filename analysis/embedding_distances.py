#!/usr/bin/env python3
"""
Compute embedding distances (H3) and base log-probabilities (H2) for all
symbol sets across epochs/modes/datasets.

H3: pairwise cosine distance between first-subword embeddings of class symbols
H2: log P(symbol) under base model given neutral prompt "The answer is: "

Usage (GPU recommended — loads full model for H2):
    conda run -n qwen python analysis/embedding_distances.py

Outputs:
    analysis/embedding_distances.tsv  — full table
"""

import re
import sys
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2AudioForConditionalGeneration

SYMBOL_FILE = os.path.join(os.path.dirname(__file__), "041203_qwen_meld_emotion_dspo_symbols.txt")
MODEL_PATH  = "Qwen/Qwen2-Audio-7B-Instruct"
NEUTRAL_PROMPT = "The answer is: "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Parse symbol file
# ---------------------------------------------------------------------------

def parse_symbol_file(path):
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
            entries.append({
                "epoch": current_epoch, "phase": current_phase,
                "mode": current_mode,   "dataset": m.group(1),
                "f1": float(m.group(2)), "symbols": {}, "tokens": {},
            })
            continue

        # "    label  → sym  2tok  ['tok1', 'tok2']"
        m = re.match(r"\s{4}(\S+)\s+→\s+(\S+)\s+\d+tok\s+\[(.+)\]", line)
        if m and entries:
            label = m.group(1)
            toks  = [t.strip().strip("'") for t in m.group(3).split(",")]
            entries[-1]["symbols"][label] = m.group(2)
            entries[-1]["tokens"][label]  = toks  # both subword tokens

    return [e for e in entries if e["symbols"]]


# ---------------------------------------------------------------------------
# H2: base log probability of a symbol under neutral prompt
# ---------------------------------------------------------------------------

def compute_base_log_probs(model, tokenizer, all_symbols: list) -> dict:
    """
    Returns {symbol: log_prob} where log_prob = log P(tok1) + log P(tok2|tok1)
    measured after "The answer is: " with no audio / no task context.
    """
    prompt_ids = tokenizer(NEUTRAL_PROMPT, return_tensors="pt",
                           add_special_tokens=True).input_ids.to(DEVICE)
    unique_symbols = sorted(set(all_symbols))
    result = {}

    model.eval()
    with torch.no_grad():
        # Single forward pass to get logits after the prompt
        out = model.language_model(input_ids=prompt_ids)
        logits_after_prompt = out.logits[0, -1, :]          # (vocab,)
        log_probs_step1 = F.log_softmax(logits_after_prompt, dim=-1)

        for sym in unique_symbols:
            tok_ids = tokenizer(sym, add_special_tokens=False).input_ids
            if len(tok_ids) < 1:
                result[sym] = float("-inf")
                continue

            lp1 = log_probs_step1[tok_ids[0]].item()

            if len(tok_ids) >= 2:
                # Feed tok1, get distribution over tok2
                extended = torch.cat([
                    prompt_ids,
                    torch.tensor([[tok_ids[0]]], device=DEVICE)
                ], dim=1)
                out2 = model.language_model(input_ids=extended)
                lp2 = F.log_softmax(out2.logits[0, -1, :], dim=-1)[tok_ids[1]].item()
                result[sym] = round(lp1 + lp2, 4)
            else:
                result[sym] = round(lp1, 4)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print("Loading tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    emb_matrix = model.get_input_embeddings().weight.detach().float()  # (vocab, hidden)
    print(f"Embedding matrix: {emb_matrix.shape}")

    entries = parse_symbol_file(SYMBOL_FILE)
    print(f"Parsed {len(entries)} symbol sets\n")

    # Collect all unique symbols for batch H2 computation
    all_syms = [sym for e in entries for sym in e["symbols"].values()]
    print(f"Computing base log-probs for {len(set(all_syms))} unique symbols...")
    base_lp = compute_base_log_probs(model, tok, all_syms)
    del model  # free VRAM after H2 done

    # ---------------------------------------------------------------------------
    # H3: embedding distances + table
    # ---------------------------------------------------------------------------
    header = (f"{'Epoch':>5} {'Phase':<14} {'Mode':<7} {'Dataset':<15} {'F1':>5}  "
              f"{'MinDist':>7} {'AvgDist':>7} {'MinPair':<25} {'AvgBaseLP':>10} {'MinBaseLP':>10}")
    print(header)
    print("-" * len(header))

    results = []
    for e in entries:
        labels     = list(e["tokens"].keys())
        first_toks = [e["tokens"][l][0] for l in labels]
        syms       = [e["symbols"][l]   for l in labels]

        ids  = [tok.convert_tokens_to_ids(t) for t in first_toks]
        vecs = torch.stack([emb_matrix[i] for i in ids])
        vecs = F.normalize(vecs.float(), dim=-1)

        sims  = vecs @ vecs.T
        dists = 1.0 - sims
        n     = len(labels)
        pairs = [(dists[i, j].item(), labels[i], labels[j])
                 for i in range(n) for j in range(i+1, n)]
        if not pairs:
            continue

        min_dist = min(p[0] for p in pairs)
        avg_dist = sum(p[0] for p in pairs) / len(pairs)
        min_pair = min(pairs, key=lambda x: x[0])

        sym_lps   = [base_lp.get(s, float("-inf")) for s in syms]
        avg_blp   = sum(sym_lps) / len(sym_lps) if sym_lps else 0
        min_blp   = min(sym_lps) if sym_lps else 0

        row = {**e, "min_dist": min_dist, "avg_dist": avg_dist,
               "min_pair": f"{min_pair[1]}∩{min_pair[2]}",
               "avg_base_lp": avg_blp, "min_base_lp": min_blp}
        results.append(row)

        print(f"{e['epoch']:>5} {e['phase']:<14} {e['mode']:<7} {e['dataset']:<15} "
              f"{e['f1']:>5.3f}  {min_dist:>7.4f} {avg_dist:>7.4f} "
              f"{min_pair[1]}∩{min_pair[2]:<20} {avg_blp:>10.3f} {min_blp:>10.3f}")

    # Correlations
    print("\n\n=== CORRELATIONS (per dataset × mode) ===")
    for ds in ["voxceleb", "hvb", "voxpopuli", "meld_emotion"]:
        for mode in ["fixed", "fresh"]:
            sub = [r for r in results if r["dataset"] == ds and r["mode"] == mode]
            if len(sub) < 3:
                continue
            f1s  = [r["f1"]          for r in sub]
            mds  = [r["min_dist"]    for r in sub]
            blps = [r["avg_base_lp"] for r in sub]

            def pearson(xs, ys):
                n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
                num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
                sx  = (sum((x-mx)**2 for x in xs)/n)**0.5
                sy  = (sum((y-my)**2 for y in ys)/n)**0.5
                return num/(n*sx*sy) if sx*sy > 0 else 0.0

            r_dist = pearson(f1s, mds)
            r_blp  = pearson(f1s, blps)
            print(f"  {ds:<15} [{mode}]  n={len(sub)}"
                  f"  H3 r(F1,min_dist)={r_dist:+.3f}"
                  f"  H2 r(F1,avg_base_lp)={r_blp:+.3f}")

    # Save TSV
    out = os.path.join(os.path.dirname(__file__), "embedding_distances.tsv")
    with open(out, "w") as f:
        f.write("epoch\tphase\tmode\tdataset\tf1\tmin_dist\tavg_dist\tmin_pair\tavg_base_lp\tmin_base_lp\n")
        for r in results:
            f.write(f"{r['epoch']}\t{r['phase']}\t{r['mode']}\t{r['dataset']}\t"
                    f"{r['f1']}\t{r['min_dist']:.6f}\t{r['avg_dist']:.6f}\t"
                    f"{r['min_pair']}\t{r['avg_base_lp']:.4f}\t{r['min_base_lp']:.4f}\n")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
