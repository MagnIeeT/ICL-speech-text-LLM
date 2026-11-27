# E5: DG-Trap variants, one change each on top of A2 (26-28 Nov)

Index: [README](README.md).

## Training (26 Nov, full data, 5 epochs, seed 1, GPUs 1-6)

Common: `TRAP_COND_SPAN=gold` (cond loss scored only on the damaged label's
tokens), validation modes `original,fresh,flipped,index`. Index mode names the
labels `label1..N` in alphabetical order. Checkpoint chosen by original-name
validation. Validation covers the trained tasks only (hvb, CREMA-D, MInDS en).

| Run | Change | Epoch | Original | Fresh symbols | Flipped | Index (label1) |
|---|---|---|---|---|---|---|
| `014933` A2 + span | baseline | 5 | 0.860 | 0.782 | 0.808 | 0.715 |
| `014938` A0 + symbol | symbol naming 1/4, no cond loss | 4 | 0.847 | 0.832 | 0.801 | **0.775** |
| `015019` A2 + symbol | symbol naming 1/4 | 5 | 0.842 | **0.836** | **0.811** | 0.726 |
| `015127` A2 + paraphrase | definitions reworded on 50% of steps | 5 | 0.844 | 0.768 | 0.787 | 0.601 |
| `015252` A2 + KL | KL to pre-trained Qwen, λ=0.1 | 5 | **0.865** | 0.753 | 0.794 | 0.567 |
| `015402` A2 + DPO | margins relative to pre-trained Qwen | 4 | 0.858 | 0.768 | 0.791 | 0.704 |

- Symbol naming is the only change that helps renamed validation (+5.4 fresh,
  hvb 0.58 to 0.66), at a cost of 1.8 points on original names.
- KL keeps the best original-name score but collapses on label1 (hvb 0.30,
  MInDS en 0.58). Paraphrase also drops. Both hurt the generic-name scenario.
- DPO changes little against the baseline.

## Test on unseen tasks (28 Nov, running)

RAVDESS, Skit, MInDS fr, MInDS ko; modes `original` (names), `fixed` with
`fixed_draw1.json` (symbols) and `fixed_generic_shuf1` (label1), one job per
checkpoint. Outputs: `metrics/2026-11-28/nt_{span,A0sym,A2sym,para,kl,dpo}/`.
