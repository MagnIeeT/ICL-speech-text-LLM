# E0 Results Log (Chapter 2 pilot)

Running record of every E0 experiment and what it showed. Append, do not
overwrite: the point is to have the full trail when we write the paper or
change direction. Plan: [`docs/chapter2_plan.md`](chapter2_plan.md).

Each section states what was run, the numbers, and what it implies. Findings
that could become paper claims are marked **[F]**.

---

## E0a: where do audio-only and transcript-only predictions diverge?

**Question.** Do the two modalities lead the model to different answers often
enough that "which evidence should decide?" is a well-posed question? If
disagreement is near zero, axis B of Chapter 2 has no headroom and the chapter
narrows to H1 and H4.

**Method.** The same checkpoint and the same items evaluated twice, changing
only `--input_mode` (`speech_only` vs `text_only`), zero-shot, canonical labels
(`--validation_modes original`), test split, 500 items per dataset. Pairing is
positional and verified against the gold-label sequence. Launcher:
`hpc/submit_e0a_disagreement.sh`; analysis: `experiments/e0a_disagreement.py`.

**Datasets.** MELD is the measurement: conversational, never in the Chapter 1
training mixture, transcripts carry real emotional content. CREMA-D is the
control: its 12 sentences are emotionally neutral ("It's eleven o'clock"), so
the transcript should carry almost nothing.

### Qwen2-Audio, 14 Nov 2026

MELD (the measurement):

| condition | checkpoint | speech | text | **disagree** | both ok | speech only | text only | both wrong |
|---|---|---|---|---|---|---|---|---|
| Instruction Tuning | `004932_qa_h-cr-m14en_nosym` ep3 | 41.8% | 52.4% | **47.6%** | 166 | 43 | 96 | 195 |
| DG-ICL-Sym | `003002_qa_h-cr-m14en_dpi_ha` ep4 | 40.4% | 50.8% | **51.6%** | 154 | 48 | 100 | 198 |
| DG-ICL-Flip | `014804_qa_h-cr-m14en_nosym_swap_per_instance` ep4 | 41.2% | 51.2% | **49.0%** | 158 | 48 | 98 | 196 |

CREMA-D (the control):

| condition | speech | text | disagree | speech only | text only |
|---|---|---|---|---|---|
| Instruction Tuning | 87.4% | 15.0% | 86.2% | 370 | 8 |
| DG-ICL-Sym | 86.2% | 14.4% | 85.8% | 365 | 6 |
| DG-ICL-Flip | 85.6% | 14.4% | 85.6% | 361 | 5 |

Pre-trained Qwen2-Audio: pending.

### What it shows

1. **The control validates the setup.** On CREMA-D the transcript carries
   nothing: text collapses to ~14% against ~86% from audio, and `speech only
   correct` beats `text only` by roughly 365 to 6. The pipeline measures what we
   intend it to.

2. **[F] Axis B has ample headroom; the kill criterion is cleared.** On MELD,
   48 to 52 percent of items yield different answers from the two modalities.
   The plan's abort condition was disagreement near zero. E0b is worth building.

3. **[F] On conversational speech these models lean lexical.** The transcript
   beats the audio by 8 to 10 points in every condition, and `text only correct`
   roughly doubles `speech only correct` (96 vs 43 for IT). This is the text
   dominance that ALME documents in GPT-4o and Gemini, now measured in an open
   model we can train.

4. **[F] Chapter 1 training did not touch modality arbitration.** The three
   conditions sit within 4 points of each other on disagreement and within 1.5
   on accuracy. This supports H2: definition-guided tuning changed behaviour
   under label-name conflict and left evidence selection essentially unchanged.
   They are separate capabilities. (Confirmation pending the pre-trained
   baseline, which distinguishes "training left it untouched" from "all three
   moved it the same way".)

---

## E0a side analysis: MELD label-surface mismatch

**Question.** MELD's inventory is `anger`/`joy`/`sadness` while CREMA-D, which
was in the training mixture, uses `angry`/`happy`/`sad`. Does that spelling gap
inflate the error rate, or contaminate the disagreement measure?

**Answer to the second part: no, the disagreement measure is clean.** Crediting
synonyms changes disagreement by at most 1.2 points, and by zero for Sym and
Flip, so E0b's evaluation subset is not an artifact of label spelling.

| cond | disagree (raw) | disagree (canonical) | surface-only items |
|---|---|---|---|
| sym | 51.6% | 51.6% | 0 |
| it | 47.6% | 46.4% | 6 |
| flip | 49.0% | 49.0% | 0 |

Accuracies move slightly, the headline does not. With synonyms credited, speech
goes 41.8 to 44.2 (IT) and 41.2 to 44.2 (Flip); Sym is unchanged at 40.4. Text
stays ahead at 50.8 / 52.8 / 52.8.

**[F] The finding worth keeping: DG-ICL-Sym respects the prompt's label
inventory and the baselines do not.** Out-of-inventory predictions on MELD
speech:

| condition | out-of-inventory | what it emits |
|---|---|---|
| DG-ICL-Sym | **0.2%** (1) | n/a |
| Instruction Tuning | 8.6% (43) | all `angry` |
| DG-ICL-Flip | 10.6% (53) | all `angry` |

IT and Flip fall back on the surface form they were trained on (`angry`, from
CREMA-D) instead of the form the legend in front of them specifies (`anger`).
DG-ICL-Sym almost never does. This is the lexical-prior shortcut in a guise
Chapter 1 never measured: not "does the model follow a contradictory
definition", but "does the model stay inside the label set it was given on an
unseen task".

This changes no Chapter 1 number, since `macro_f1_with_invalid` already counts
out-of-inventory predictions as wrong. It was simply never reported as a
phenomenon. Candidate table for Chapter 2, or supporting evidence if Chapter 1
reviewers ask for more on instruction following.

---

## Pending

- Pre-trained Qwen2-Audio E0a (running 14 Nov). Distinguishes "Chapter 1
  training left arbitration untouched" from "all three conditions moved it
  identically".
- E0a on Audio Flamingo 3. AF3 fuses audio by cross-attention rather than token
  concatenation, so arbitration may differ. If axis B is Qwen-specific the
  chapter changes, so this gates E0b rather than following it.
- E0b, the routing probe: `speech_text` input mode plus routed legends, measured
  on the disagreement subsets saved at `~/training/e0a/disagreement_*.json`.

## Artifacts

| What | Where |
|---|---|
| Prediction dumps | `~/training/e0a/metrics/<date>/<run>/<run>_p.json` |
| Disagreement subsets | `~/training/e0a/disagreement_{sym,it,flip}.json` |
| Run logs | `~/training/e0a/logs/` |
| Inference logs | `~/training/symbol_training/logs_inference/<date>/` |

Not in git: all of the above live outside the repo. Only the scripts and this
log are versioned.
