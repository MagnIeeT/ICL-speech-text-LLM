# E3: DG-Trap A0-A3 (22-25 Nov)

Index: [README](README.md).

## E3: DG-Trap, definition-trap training (A0-A3 trained 22 Nov, tested 23 Nov; results below)

**Question.** Can one model decide from the definitions whatever the labels are
called (original words, swapped words, symbols, `label1..N`), instead of each
method being best only in its own training format? Method and motivation:
`docs/chapter2_plan.md` section 13a.

**Method (one line per part).** Real label words only, batch size 1, single stage
from pretrained, no reference model.
- Naming per instance: `orig` / `goldswap` (gold swaps names with one non-gold;
  partner = nearest-neighbour definition half the time, else random) /
  `pairswap` (two non-gold swap; blocks the "odd line = answer" leak). One third each.
- Harmless dropout: non-deciding words of random legend lines (gold included)
  dropped at a rate ~ U(0, 0.2) per instance.
- `L = NLL + lambda_t * L_trap + lambda_c * L_cond`.
  `L_trap = -log sigmoid(beta * (log p(gold tok) - log p(trap tok)) - margin)` at the
  first completion token where the shown gold name and the trap (gold's real
  name) differ; same pass. `L_cond` compares log p(y) with the full legend vs the
  legend with the gold line's deciding words removed (second pass, 30% of
  instances). beta 2, margins 0.5, lambdas 0.5.

**Deciding words and neighbours.** `scripts/build_deciding_words.py` ->
`config/data_config/deciding_words.json`. Neighbour = most similar definition by
`BAAI/llm-embedder` CLS cosine (automated; overrides file supported but empty).
Deciding words = definition words not in the neighbour's definition (function
words removed; custom list, scikit-learn's list wrongly drops "bill").

**Code.** `models/symbolAdapter/trap_loss.py` (pure functions),
`symbol_training.py` (`_init_trap`, `_trap_step`, dispatch in `_train_one_epoch`),
`training_configs.py` (`trap_*` fields and flags), `validation.py` (new `index`
mode: `label1..labelK` in sorted-label order, a format no method trains on),
`hpc/submit_symbol_training_node1.sh` (`TRAP_*` env vars, run tag
`trap_lt<l_t>_lc<l_c>[_nopair][_condrandom]`).

**Logit alignment.** Qwen2-Audio (transformers 4.45) expands `<|AUDIO|>` into the
audio frames inside `forward`, so `logits` is longer than `input_ids` and
`prompt_length` does not index it. DG-Trap locates completion rows from the end
of the (unpadded, batch 1) sequence and, every step, recomputes the NLL from
those rows and compares it with the model's loss (`align_err` in the logs).
**Side finding:** E2's `definition_log_probs(logits, prompt_length, ...)` indexes
the expanded logits with the unexpanded prompt length, so the E2 divergence
was read from a row inside the prompt, not the answer row (reproduced in
`tests/test_dg_trap.py` with a mock that expands like Qwen2-Audio; not yet
confirmed on GPU). This may explain "the coupling is not what helps" in E2.

**Tests.** `HF_HUB_OFFLINE=1 python tests/test_dg_trap.py` (CPU, real tokenizer,
real prompts from an IT training log in `tests/fixtures/trap_prompts.json`,
real renaming code, mock model with audio expansion): 15/15 pass. Covers text
edits, index coverage of every legend line, goldswap renaming, end-aligned rows
(err < 1e-5), shared-prefix trap token (`question_general` vs `question_check`,
k = 1), 240 end-to-end steps incl. multi-label HVB, view construction (only the
gold line damaged, deciding words never dropped), A0 = pure NLL with one pass,
naming proportions, cond_mode=random, guards, `index` mode round trip, plus
three regression tests from an independent code review (22 Nov): neighbour
and deciding words follow the prompt's own wording variant; text input modes
and few-shot prompts are rejected (renaming would rewrite label words inside
transcripts); a non-finite trap or conditional term is skipped, not trained.
Known nits left: the run tag does not encode beta/margin/naming-proportion
changes, and a resumed run re-seeds the naming RNG.

**Runs (each 5 epochs, full data, seed 1, `KEEP_ALL_CHECKPOINTS=true`; report the
epoch chosen by trained-task validation, as in the paper).**

| Run | Env (on top of `TRAP_ENABLED=true`) | Purpose |
|---|---|---|
| A0 | `TRAP_LAMBDA=0 TRAP_COND_LAMBDA=0` | data only (naming + dropout); key baseline |
| A1 | `TRAP_COND_LAMBDA=0` | + trap margin |
| A2 | `TRAP_LAMBDA=0` | + conditional margin |
| A3 | defaults | full method |
| A4 | `TRAP_P_PAIRSWAP=0` | does the leak blocker matter |
| A5 | `TRAP_COND_MODE=random` | do deciding words matter |

Start with A0 and A3 (two GPUs). Launch template (ask before running):

```
ssh node1 "cd /home/aneeraj/ICL-speech-text-LLM && TRAP_ENABLED=true MODEL_TYPE=qwen \
  DATASET_TYPE=hvb-cremad-minds14_en MAX_SAMPLES=0 LORA_EPOCHS=5 SEED=1 NUM_EXAMPLES=0 \
  VALIDATION_MODES=original,fresh KEEP_ALL_CHECKPOINTS=true \
  CUDA_VISIBLE_DEVICES=<gpu> bash hpc/submit_symbol_training_node1.sh"
```

Training-time validation keeps the earlier runs' modes (`original,fresh`, trained
tasks) so checkpoint selection matches the paper; `flipped` and `index` are used
at test evaluation only.

**Evaluation.** Held-out test (Skit, RAVDESS-Song, MInDS fr/ko) in four formats:
`original,fresh,flipped,index`; plus the gold-definition diagnostic and the
confusable pairs (`branch_address`/`ifsc_code`, `sad`/`calm`/`fearful`).
Success = the four columns converge near the best current value (about 0.8)
from one checkpoint, and A3 beats A0.

**What to watch in the training log.**
- `DG-Trap ON: ...` at start: settings and datasets in the index.
- `TRAP-DETAIL` (batches 0-1 of each epoch, then every 1000): kind, mapping,
  partner source, shown completion, trap tokens and position k, removed words,
  every loss term. The full prompts of both views are logged just before it.
- `TRAP-RUN` (every 200 batches) and `TRAP-EPOCH`: kind counts, nll, trap
  loss / diff / win rate (share with gold token above trap token), cond loss /
  diff / win rate, dropped and removed words per step, `align_err_max`, skip
  counts, `unmatched_lines`.
- Healthy: `align_err_max` below 1e-2 (else a warning fires), `unmatched_lines=0`,
  kinds near one third each, trap win rate rising toward 1, `failed_batches=0`
  (more than 20% failed batches aborts the run with tracebacks logged).

### Changes after the first review (22-23 Nov)

- E2 fix: `definition_log_probs(..., input_len=)` shifts the answer row by the
  audio expansion; consistency now requires batch size 1. Test `test_e2_fixed_row`.
- Empty damaged line: when removing the gold line's deciding words would leave it
  empty (17 of 152 definitions, mostly terse wording; hvb `backchannel` in orig),
  the conditional term is skipped and counted as `skip_cond_empty`. An empty line
  never occurs in normal prompts, so it would be a shortcut. Test
  `test_cond_empty_skipped`. Full suite 16/16 before this change, new test plus
  related tests pass after it.
- `hpc/submit_symbol_inference_test.sh`: default `MODEL_TYPE=qwen`; the
  hard-coded SALMONN checkpoint line (old line 244) is now a comment; a guard
  refuses a checkpoint whose run tag (`_qa_`/`_sl_`/`_af_`) does not match
  `MODEL_TYPE`. The first 23 Nov test runs loaded that SALMONN checkpoint and are
  kept as `logs_inference/2026-11-23/dgtrap_*_INVALID_wrongckpt.log`.

### Smoke test (22 Nov, GPU 1)

300 samples per task, 1 epoch, full method. Real-model `align_err_max` 4.8e-07,
0 failures, 0 skips, kinds 316/288/296, partners nn/random 149/139. Trap win
rate 0.19 -> 0.25 within the epoch; trap loss (7.2) was 4x the NLL (1.8).

### Training (22 Nov, full data, 5 epochs, seed 1)

| Run | GPU | Log (`logs/2026-11-22/`) |
|---|---|---|
| A0 | 1 | `153525_qa_h-cr-m14en_trap_lt0_lc0` |
| A1 | 2 | `153529_qa_h-cr-m14en_trap_lt0.5_lc0` |
| A2 | 3 | `153534_qa_h-cr-m14en_trap_lt0_lc0.5` |
| A3 | 4 | `153837_qa_h-cr-m14en_trap_lt0.5_lc0.5` |

All finished, 0 failed batches, `align_err_max` at or below 1e-6. Epoch 1
summaries: trap win rate A0 0.81 (measured, not trained), A1 0.85, A3 0.86;
cond win rate A2 0.96, A3 0.96. Renaming alone already teaches most of the trap.

Validation (trained tasks, 500 per task, macro F1 average):

| Epoch | A0 orig | A1 orig | A2 orig | A3 orig | A0 fresh | A1 fresh | A2 fresh | A3 fresh |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.822 | 0.817 | 0.817 | 0.807 | 0.744 | 0.720 | 0.715 | 0.723 |
| 2 | 0.835 | 0.836 | 0.826 | 0.832 | 0.777 | 0.779 | 0.786 | 0.786 |
| 3 | 0.851 | 0.853 | 0.834 | 0.836 | 0.782 | 0.776 | 0.775 | 0.779 |
| 4 | 0.860 | **0.868** | **0.856** | 0.840 | 0.793 | 0.781 | 0.776 | 0.770 |
| 5 | **0.867** | 0.866 | 0.856 | **0.841** | 0.799 | 0.789 | 0.798 | 0.791 |

No early peak: original keeps rising to epoch 5. Selected by original
validation: A0 ep5, A1 ep4, A2 ep4, A3 ep5.

### Test results (23 Nov)

Test split, full test sets, speech only, zero-shot, orig definitions,
`original,fresh`, one GPU per model. Logs
`logs_inference/2026-11-23/dgtrap_A{0,1,2,3}_ep{5,4,4,5}_test.log`. IT, Sym and
Flip are the paper's 3-seed means (`tables/appendix_tables/per_task_qwen.tex`);
A0-A3 are one seed. Scoring is the old rule (a real name not shown in a renamed
prompt still counts), the same as the paper numbers.

Original labels, macro F1:

| Task | IT | Sym | Flip | A0 | A1 | A2 | A3 |
|---|---|---|---|---|---|---|---|
| HVB | 0.674 | 0.572 | 0.655 | 0.687 | 0.683 | 0.673 | 0.681 |
| CREMA-D | 0.867 | 0.832 | 0.854 | 0.861 | 0.867 | 0.858 | 0.869 |
| MInDS en | 0.974 | 0.865 | 0.973 | 0.970 | 0.964 | 0.972 | 0.963 |
| **Trained avg** | 0.838 | 0.756 | 0.827 | 0.839 | 0.838 | 0.834 | 0.838 |
| Skit-S2I | 0.851 | 0.798 | 0.762 | 0.795 | 0.791 | 0.811 | 0.756 |
| RAVDESS-Song | 0.660 | 0.520 | 0.508 | 0.594 | 0.545 | 0.573 | 0.553 |
| MInDS fr | 0.848 | 0.765 | 0.878 | 0.878 | 0.869 | 0.840 | 0.840 |
| MInDS ko | 0.531 | 0.451 | 0.522 | 0.514 | 0.575 | 0.529 | 0.558 |
| **Unseen avg** | **0.723** | 0.634 | 0.668 | 0.695 | 0.695 | 0.688 | 0.677 |

Fresh symbols, macro F1 (each run draws its own symbols):

| Task | IT | Sym | Flip | A0 | A1 | A2 | A3 |
|---|---|---|---|---|---|---|---|
| HVB | 0.314 | 0.636 | 0.594 | 0.513 | 0.519 | 0.576 | 0.537 |
| CREMA-D | 0.718 | 0.851 | 0.854 | 0.839 | 0.846 | 0.854 | 0.841 |
| MInDS en | 0.877 | 0.974 | 0.962 | 0.834 | 0.948 | 0.883 | 0.970 |
| **Trained avg** | 0.636 | **0.820** | 0.803 | 0.729 | 0.771 | 0.771 | 0.783 |
| Skit-S2I | 0.810 | 0.767 | 0.803 | 0.753 | 0.803 | 0.826 | 0.720 |
| RAVDESS-Song | 0.484 | 0.668 | 0.645 | 0.588 | 0.640 | 0.649 | 0.502 |
| MInDS fr | 0.743 | 0.856 | 0.866 | 0.856 | 0.867 | 0.815 | 0.827 |
| MInDS ko | 0.443 | 0.521 | 0.454 | 0.436 | 0.343 | 0.478 | 0.453 |
| **Unseen avg** | 0.620 | **0.703** | 0.692 | 0.658 | 0.663 | 0.692 | 0.626 |

Unseen average, both formats:

| Model | Original | Fresh | Worse of the two |
|---|---|---|---|
| IT | **0.723** | 0.620 | 0.620 |
| Sym | 0.634 | **0.703** | 0.634 |
| Flip | 0.668 | 0.692 | 0.668 |
| A0 (data only) | 0.695 | 0.658 | 0.658 |
| A1 (+ trap) | 0.695 | 0.663 | 0.663 |
| A2 (+ cond) | 0.688 | 0.692 | **0.688** |
| A3 (both) | 0.677 | 0.626 | 0.626 |

Reading:

- Original, trained tasks: all four reach IT (0.84); Sym loses 0.08.
- Original, unseen: A0-A2 sit between Flip and IT (0.03 below IT).
- Fresh, unseen: only A2 reaches Flip (0.692) and comes within 0.011 of Sym.
- A2 is the most balanced (worst format 0.688), but it is **about the same as
  Flip**: +0.020 in original, tie in fresh, -0.032 on trained tasks in fresh.
  One seed against 3-seed means, so DG-Trap is not yet shown to beat Flip.
- The trap loss does not help on unseen tasks; the full method (A3) is the
  weakest on unseen tasks in both formats.

Caveats: one seed; each run drew its own fresh symbols (single draws can move a
task by up to 20 points, e.g. A0 MInDS en 0.834 vs 0.95-0.97 for the others).

Next: one fixed symbol file for all models (`fixed` mode + `SYMBOL_MAP_FILE`),
IT and Sym re-evaluated on the same prompts, the index scoring fix, flipped and
index formats, the clustering eval (3 naming styles), and more seeds for A2.

### Analysis of the A0-A3 runs (24 Nov, logs and saved predictions, no GPU)

Question: for more gain, should we raise lambda? Four checks, all from existing
files. Scripts: `err_breakdown.py`, `nn_errors.py`, `paired.py` (session
scratchpad). Baselines are single runs: IT `004932` ep3, Sym `003002` ep3, Flip
`014804` ep4. Accuracy below is exact match, so hvb differs slightly from the
official metric.

**1. Both extra losses are saturated after epoch 1, so lambda is not the lever.**

| Epoch | Trap win A0 (measured only) | Trap win A1 (trained) | Cond win A2 | Cond loss A2 | Cond gap A2 (margin 0.5) |
|---|---|---|---|---|---|
| 1 | 0.811 | 0.850 | 0.963 | 0.187 | 3.7 |
| 2 | 0.966 | 0.970 | 0.995 | 0.025 | 6.5 |
| 5 | 0.978 | 0.980 | 0.998 | 0.009 | 9.5 |

- Trap is redundant with NLL: A0 never trains it and still reaches 0.978. NLL on
  goldswap rows already raises the correct name and so lowers the trap name.
- Cond is solved by epoch 2. A larger lambda multiplies a near-zero loss.
- Both checks run on the same 3 training tasks with fixed definition text every
  epoch, so the model likely memorises them.
- A0 does not measure cond, so we do not know whether NLL alone also gets there.

**2. Format is solved; the gap is choosing the label.** Unseen tasks, fresh,
percent of items, task average:

| Model | Correct | Invalid | Real name instead of symbol | Wrong but valid |
|---|---|---|---|---|
| IT | 50.9 (+14.1 via the scoring leak) | 4.4 | 6.7 | 23.9 |
| Sym | **75.1** | 1.5 | 0 | 23.4 |
| Flip | 69.3 | 13.4 | 0 | **17.4** |
| A0 | 70.1 | 1.5 | 0 | 28.4 |
| A2 | 71.9 | **0.4** | 0 | 27.7 |

- A2 almost never gives an invalid output and never falls back to a real name.
- Among valid outputs it is right 72% of the time, against Sym 76% and Flip 80%.
- On trained tasks (fresh), A2 has 4.6% invalid against Sym 0.3%. These are copy
  errors of random strings (`prmhnz` for `prmhn`, 303 times on hvb in A0), so they
  depend on the symbol draw.
- IT's fresh score is inflated by the leak: 63% of its CREMA-D "correct" answers
  are real names not shown in the prompt.

**3. The cond loss removes neighbour confusions, but most remaining errors are
not between similar definitions.** Neighbour = most similar definition by
`BAAI/llm-embedder` (same script as training; unseen-task neighbours computed now,
not hand-reviewed). Single-label tasks only. Unseen, fresh:

| Model | Errors that go to the gold's neighbour | Chance | Errors that go to the gold's most common wrong label |
|---|---|---|---|
| A0 | 35.6% | 10% | 69% |
| **A2** | **18.8%** | 10% | 62% |
| Sym | 21.8% | 10% | 63% |
| Flip | 26.6% | 8% | 55% |
| IT | 33.1% | 10% | 69% |

- Neighbour errors halve from A0 to A2 (Skit 47.7% to 28.6%), the lowest of all
  models. The cond loss transfers to new tasks.
- Outside Skit, A2's neighbour rate is at or below chance (RAVDESS 5%, MInDS fr
  0%, MInDS ko 3.5%), yet about 60% of each gold's errors go to one label.
- A2 top confusions: RAVDESS fearful to sad 28, neutral to sad 12, calm to sad 10;
  MInDS ko freeze to card_issues 6, cash_deposit to atm_limit 5; Skit
  branch_address to ifsc_code 46 (a neighbour), card_issue to
  unauthorised_transaction 18.
- These systematic errors come from the audio or from label bias, not from close
  definitions, so losses on the definition text cannot reach them.

**3b. A2 does not lean on label names more than Sym.** Same model, same item,
original vs fresh, share of items with the same decided label (unseen,
single-label): IT 81.1%, Sym 77.3%, Flip 75.2%, A0 82.8%, **A2 83.3%**. A2 is the
least sensitive to the name form. Its fresh gap to Sym is mostly format
familiarity: Sym trained on random strings, A2 never saw them (A2 trails Sym by
12.8 on hvb and 9.7 on MInDS en, largely copy errors). On unseen tasks, item by
item, only-Sym-right 139 vs only-A2-right 110 of 1889 (about 1.5 points, McNemar
p about 0.07). With real names A2 beats Sym 150 to 106 on unseen items.

**4. A2 vs A0, item by item (effect of the cond loss).**

| Task | Original diff | Fresh diff | Fresh: only A2 right / only A0 right |
|---|---|---|---|
| hvb (T) | -0.7 | **+4.3** | 689 / 426 |
| CREMA-D (T) | -0.4 | +1.4 | 28 / 18 |
| MInDS en (T) | +0.3 | **+5.2** | 28 / 9 |
| RAVDESS (U) | -3.0 | +3.0 | 28 / 20 |
| Skit (U) | +0.7 | **+5.4** | 135 / 60 |
| MInDS fr (U) | -1.9 | -3.8 | 3 / 7 |
| MInDS ko (U) | +0.8 | +2.5 | 10 / 7 |

With symbols the cond loss wins on 6 of 7 tasks (Skit and hvb far beyond noise);
with real names it costs nothing beyond a few items.

**The RAVDESS "sad" sink.** Gold has 48 sad items. Predicted sad: A2 94 (original)
and 92 (fresh, string `nevh`); A0 66 and 100 (`ligak`); IT 94 and 102; Sym 0 and 77
(`gsxt`). Sym and Flip in original mode instead sink to fearful (118, 140). Within
one run each label has one symbol for all items, so a fresh run is a single draw.
A2 picks sad as often with the word "sad" as with `nevh`, which suggested the pull
follows the sad definition ("Sorrow or disappointment") and the audio. The
follow-up below confirms the direction but shows the size depends on the string. Calm ("A relaxed, peaceful state") is almost never predicted with real
names by IT, A0 and A2 (0, 0, 2).

**Conclusions.**
- Do not tune lambda; the losses are silent after epoch 1.
- Drop the trap loss as a separate term; NLL on renamed rows already does it.
- Keep the cond loss: it gives a real gain under symbols at no cost with names.
- Remaining levers:
  - a random-symbol naming kind in training, to close the fresh-format gap to
    Sym (open: briefly dropped on 25 Nov, but that evidence used malformed
    symbol strings; see the correction under "Sym on the same symbol files");
  - trap partners chosen from the model's own confusions instead of text
    similarity, to reach the systematic errors (fearful to sad);
  - label-bias correction for sinks like "sad";
  - paraphrased definitions, so the training checks stay hard.

### Sad-sink follow-up: A2 on RAVDESS with fixed symbols and swapped definitions (24 Nov)

Setup: A2 (`153534` ep4), RAVDESS test only (264 items; gold 48 per label, 24
neutral), `fixed` mode with hand-written symbol files so every string is known.

- R1: symbols A (`~/training/symbol_training/symbol_maps/ravdess_A.json`),
  original definitions.
- R2: symbols B (`ravdess_B.json`), original definitions.
- R3: symbols A plus `DEFINITION_VARIANT=swap_sad_calm`, a new variant in
  `config/data_config/ravdess_song_config.py`. Sad and calm swap definitions;
  names and line order are unchanged. Run in original and fixed modes.

Logs: `logs_inference/2026-11-24/sadsink_A2_R{1,2,3}*.log`. All three loaded Qwen
and the A2 checkpoint, logged `Symbol map overridden`, and produced only strings
from their file. Script: `sadsink.py` (session scratchpad).

Picks per line:

| Run | "Sorrow or disappointment" line | "A relaxed, peaceful state" line | Fearful line | Accuracy |
|---|---|---|---|---|
| 23 Nov test, fresh draw | 92 | 42 | 16 | 0.705 |
| R1, symbols A | 62 (`srbe`) | 40 (`topf`) | 56 (`siccq`) | 0.712 |
| R2, symbols B | 86 (`ajbea`) | 30 (`hzisv`) | 12 (`mxtqu`) | 0.659 |
| R3, symbols A, swapped | **144** (`topf`) | **0** (`srbe`) | 16 | 0.576 |
| R3, real names, swapped | **102** (name "calm") | **14** (name "sad") | 30 | 0.591 |

Findings:

1. **A2 follows the definition, not the line, string or name.** With the Sorrow
   text on the calm line, all 48 sad items follow it (sad to calm 48) in both
   formats. With real names this is a direct contest, and the definition beats the
   name "sad". A2 shows the intended DG behaviour on an unseen task.
2. **The "Sorrow or disappointment" text is a sink on this audio.** It always gets
   more than its 48 gold items (62 to 144) and pulls in fearful and neutral items
   (R3 fixed: fearful 32, neutral 20). So the RAVDESS "sad" sink comes from the
   definition matched against the audio, not from the label name.
3. **The random string changes the size of the effect.** Same model, audio and
   definitions: fearful line 56 picks with `siccq` vs 12 with `mxtqu`; the sad sink
   ranges 62 to 92; accuracy 0.712 vs 0.659. A single fresh draw carries about
   +-5 points of noise per task, which is larger than the A2 vs Sym vs Flip gaps.
   *Caveat (25 Nov): files A and B break the 2-token rule of fresh mode (see the
   correction in the next section), so the size of this spread is not
   representative of normal draws. Findings 1 and 2 are unaffected.*

Implications:

- Symbol-mode comparisons need one shared symbol file per draw for all models,
  averaged over several draws.
- The real-name swap is a direct name-vs-definition probe; running it for IT, Sym
  and Flip would show which models follow the definition.

### Sym on the same symbol files (25 Nov)

> **Correction (25 Nov evening): the A/B symbol files were malformed, so the
> string-sensitivity comparison below is unreliable.** Fresh mode only uses
> strings that tokenize to exactly 2 Qwen tokens and stay intact next to commas
> (`generate_fresh_symbols`). The hand-made `ravdess_A`/`ravdess_B` strings are
> 2 to 4 tokens, and some split differently after a comma (A: `vabv` 3, `nyshc`
> 3, `siccq` 3, `jayzw` 2 alone but 4 after ", "; B: `qyozm` 4, `hzisv` 3 alone
> but 5 after ", ", the rest 3). Sym trained only on 2-token strings and A2 on
> none, so both saw out-of-format strings, and Sym's 108-pick `qyozm` sink is a
> 4-token string. Not supported any more: "Sym is more string-sensitive than A2",
> "training on random strings does not remove string preference", the 5.3 vs
> 13.6 point spreads, and the A2/Sym tie over the two draws. Still valid: the
> definition-swap result (R3, same strings in both arms, and also with real
> names), and the general need for shared, averaged symbol draws. Side
> observation only: odd strings (e.g. 4 tokens) can create large sinks.
> Replacement: three proper files, `symbol_maps/fixed_draw{1,2,3}.json` (seeds
> 2101-2103, made with `generate_fresh_symbols`, all 7 test tasks), to be run on
> every checkpoint.

Question: is string preference specific to A2, and would training on random
strings (as Sym does) remove it? Sym `003002` ep3, RAVDESS test, `fixed` mode with
the same `ravdess_A`/`ravdess_B` files. Logs:
`logs_inference/2026-11-25/sadsink_Sym_R{1,2}*.log`. Both runs loaded Qwen and the
Sym checkpoint, logged `Symbol map overridden`, and produced only strings from
their file.

| | A2, set A | A2, set B | Sym, set A | Sym, set B |
|---|---|---|---|---|
| Accuracy | 0.712 | 0.659 | **0.765** | **0.629** |
| Sorrow (sad) line | 62 | 86 | 50 | **6** (`ajbea`) |
| Angry line | 46 | 46 | 44 | **108** (`qyozm`) |
| Fearful line | 56 | 12 | 56 | 40 |
| Neutral line | 2 | 16 | 2 | 38 |

Findings:

1. **Sym is more string-sensitive than A2.** Accuracy moves 13.6 points between
   the two sets for Sym, 5.3 for A2. The summed change in per-line pick counts is
   200 for Sym and 108 for A2.
2. **Sym's errors follow the strings; A2's follow the content.** With set B,
   `qyozm` (angry) becomes a 108-pick sink for Sym and the sad line nearly
   vanishes (sad to angry 34). A2's main pull stays on the "Sorrow or
   disappointment" text under both sets.
3. **Training on random strings does not remove string preference.** Sym saw a new
   random string almost every step and is still the most sensitive, so a
   random-symbol naming kind in DGPO is not expected to fix it. Dropped from the
   lever list.
4. **Averaged over the two draws, A2 and Sym tie** (A2 0.686, Sym 0.697). One draw
   makes Sym look 5 points ahead (set A) or 3 points behind (set B).

Implications:

- Symbol-mode evaluation must use the same symbol files for every model and
  average over several draws (at least 3 to 5). Single fresh draws, as in the
  23 Nov test tables, cannot rank A2, Sym and Flip.
- A2 depends more on the definitions and less on the surface form than Sym,
  which supports the DGPO direction.

### A2 vs Sym on shared fixed symbol draws (25 Nov)

Setup: A2 seeds 1-3 (`153534` ep4, `024624` ep5, `024628` ep5) and Sym seeds 1-3
(`003002` ep3, `003016` ep2, `003227` ep3), test split, `fixed` mode with
`symbol_maps/fixed_draw1.json` and `fixed_draw2.json` (proper 2-token strings), so
6 runs per model. Draw-1 runs also ran `original`. Draw 3, IT and Flip not run yet;
hvb skipped. Outputs: `metrics/2026-11-25/fx_{A2,Sym}_s{1,2,3}_d{1,2}/`. All 12
runs finished; every fixed-mode output was checked against its map. IT numbers
below are the paper's 3-seed original results.

Macro F1 (with invalid):

| | IT (original) | A2 original | Sym original | A2 symbols | Sym symbols |
|---|---|---|---|---|---|
| RAVDESS | 0.660 | 0.558 | 0.518 | 0.498 | **0.570** |
| Skit | 0.851 | 0.802 | 0.798 | **0.787** | 0.764 |
| MInDS fr | 0.848 | 0.850 | 0.765 | 0.807 | 0.800 |
| MInDS ko | 0.531 | 0.527 | 0.449 | 0.405 | **0.472** |
| **Unseen avg** | **0.723** | 0.684 ±0.007 | 0.633 ±0.006 | 0.624 ±0.023 | **0.652** ±0.015 |
| CREMA-D | 0.867 | 0.856 | 0.829 | 0.753 | 0.806 |
| MInDS en | 0.974 | 0.970 | 0.865 | 0.935 | 0.964 |
| Trained avg | 0.921 | 0.913 | 0.847 | 0.844 | 0.885 |

Unseen avg per draw: A2 0.639 (d1), 0.610 (d2); Sym 0.659 (d1), 0.644 (d2).

Findings:

1. **With real names A2 beats Sym by 5 points but trails IT by 4.** The IT gap is
   mostly RAVDESS (-10, the fearful/Sorrow issue above) and Skit (-5).
2. **With symbols Sym leads A2 by 2.8 points on unseen tasks**, driven by RAVDESS
   (-7.2) and MInDS ko (-6.7). A2 wins Skit (+2.3) and MInDS fr (+0.7).
3. **About a third of that gap is copying.** Exact-match accuracy on unseen tasks:
   A2 0.660, Sym 0.688 (gap 2.8). Mapping near-miss outputs to the intended
   symbol: A2 0.699, Sym 0.717 (gap 1.8). The remaining 1.8 points are wrong
   label choices.

Copy errors (output not in the map, attributed to the map symbol with the same
first token; script `copy_errors.py`):

| | A2 | Sym |
|---|---|---|
| Miscopied outputs | 4.9% | 1.2% |
| Second token common (BPE id < 20k), 82 symbols | **0.0%** (1/11462) | 0.9% |
| Second token rare (BPE id >= 20k), 53 symbols | **13.5%** (865/6387) | 1.6% |
| First token 1 char / 2 chars / 3+ chars | 20.5% / 4.2% / 2.2% | about 11% / low / low |

- A2's miscopies are almost entirely on symbols whose second token is a rare
  word piece. The first token is copied, the second drifts to a more common
  piece. Worst: `ueivr`, `iiasi`, `qkek`, `vnqe`, `cliqm`.
- Sym copies random strings well (it trained on them). Its main failures are
  1-character first tokens and `iiasi` (`i`+`iasi`, 82/90).
- Per-symbol miscopy rates correlate weakly between A2 and Sym (r=0.37), so
  copyability is partly model specific.

Implications:

- A2's symbol-mode deficit is partly a copy skill A2 never practised, not only
  definition following. Copy practice (a random 2-token naming kind) is a
  training lever.
- Copyability can be predicted from the tokenizer alone, with no labels or model
  runs. Any symbol-based comparison should report it, and IT/Flip should be run on
  the same draws before ranking all four models.

