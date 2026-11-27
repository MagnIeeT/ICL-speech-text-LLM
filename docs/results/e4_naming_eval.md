# E4: naming and definition tests, strict rescore (26 Nov)

Index: [README](README.md).

### RAVDESS with paraphrased definitions (26 Nov)

Setup: IT and A2 seeds 1-3, RAVDESS test split, definitions rewritten so the
deciding words no longer echo the label names. Labels shown either as names or as
`fixed_draw1.json` symbols. Outputs: `metrics/2026-11-2[56]/rp_{A2,IT}_s{1,2,3}/`,
script `rav_para.py`.

| Model | Labels | Definitions | Macro F1 | Fearful recall | fearful->sad | sad picks |
|---|---|---|---|---|---|---|
| IT | names | original | **0.660** ±0.006 | 0.89 | 5.3 | 92.0 |
| IT | names | paraphrased | 0.651 ±0.002 | 0.82 | 8.7 | 96.0 |
| IT | symbols | paraphrased | 0.549 ±0.071 | 0.31 | 10.0 | 71.3 |
| A2 | names | original | 0.558 ±0.013 | 0.58 | 16.0 | 93.3 |
| A2 | names | paraphrased | 0.572 ±0.024 | 0.75 | 7.3 | 46.0 |
| A2 | symbols | original | 0.517 ±0.048 | 0.39 | 18.0 | 70.7 |
| A2 | symbols | paraphrased | 0.551 ±0.040 | 0.61 | 8.7 | 32.0 |

Findings:

1. **Paraphrasing fixes A2's fearful->sad sink** (16.0 to 7.3 errors, sad picks
   93 to 46) but macro F1 rises only 1.4 points: the errors move to other labels.
2. **IT ignores the definitions.** Its names score barely moves when they are
   rewritten, and it drops 11 points when names are hidden. It classifies RAVDESS
   from the label names.
3. **calm recall is 0 for every model**, including the pre-trained one (below).

### Generic label names, label1..N (26 Nov)

Setup: labels replaced by `label1..labelN`, shuffled per task
(`symbol_maps/generic_shuf1.json`), strict scorer. IT and A2 seeds 1-3, Sym seed
1. Outputs: `metrics/2026-11-26/gl_*_g1/`, script `generic_cmp.py`. Unseen-task
averages:

| Model | Names | label1 | Symbols | label1 outputs off the list (of 2989) |
|---|---|---|---|---|
| IT | **0.722** | 0.591 | n/a | 775 (26%) |
| A2 | 0.684 | 0.549 | 0.624 | 279 (9%) |
| Sym (seed 1) | 0.627 | **0.692** | 0.658 | 52 (2%) |

Findings:

1. **IT collapses under generic names**: a quarter of its outputs are trained
   label names that were not on the list.
2. **A2 is label-invariant on emotion** (RAVDESS 0.57 names vs 0.56 label1, seed
   1) but loses up to 23 points on intent tasks, mostly format failures.
3. **Sym is the only model that is better with label1 than with names.** It has
   seen arbitrary names in training; A2 has seen only real names, swapped.

### Strict rescore of the paper's Qwen symbolic numbers (26 Nov)

The scorer leak (an unshown original name accepted as correct in renamed modes,
fixed in `validation.py` with `_uses_shown_names`) inflated only models that
answer with native names:

| Method | Held-out avg (reported -> strict) | Trained avg | RAVDESS |
|---|---|---|---|
| Pre-trained | 0.467 -> 0.316 | 0.368 -> 0.177 | 0.592 -> 0.000 |
| IT | 0.620 -> 0.501 | 0.636 -> 0.442 | 0.485 -> 0.009 |

Symbol Tuning, Flipped-Label Tuning, Flip (ours) and Sym (ours) do not change
(0.349, 0.657, 0.692, 0.703). The submitted paper therefore understates our
margin over the baselines; nothing it claims for our methods depends on the leak.

### What pre-trained Qwen knows about RAVDESS (26 Nov)

- With names, pre-trained Qwen recalls happy, sad and angry perfectly (1.00) but
  never predicts calm, the RAVDESS-specific label.
- With symbols it ignores the legend and answers native emotion names, which is
  why its strict RAVDESS symbol score is 0.000.
- So RAVDESS-like emotion knowledge is in the base model, but there is no
  evidence of RAVDESS label contamination: a model trained on RAVDESS would know
  calm.
- Implication: the trained models (A2, Sym, IT) lose part of that native name
  knowledge. Two training changes test this: KL to pre-trained Qwen on
  orig-named instances, and DPO-style margins against the pre-trained model.

