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
| **Pre-trained** | no LoRA | **29.2%** | **28.8%** | **75.8%** | 81 | 65 | 63 | 291 |

CREMA-D (the control):

| condition | speech | text | disagree | speech only | text only |
|---|---|---|---|---|---|
| Instruction Tuning | 87.4% | 15.0% | 86.2% | 370 | 8 |
| DG-ICL-Sym | 86.2% | 14.4% | 85.8% | 365 | 6 |
| DG-ICL-Flip | 85.6% | 14.4% | 85.6% | 361 | 5 |
| Pre-trained | 55.4% | 9.4% | 86.2% | 236 | 6 |


### Audio Flamingo 3, 14 Nov 2026

Same 3-task mixture, same epoch 4, same items.

MELD:

| condition | checkpoint | speech | text | text - speech | **disagree** | speech only | text only |
|---|---|---|---|---|---|---|---|
| Instruction Tuning | `112544_af_h-cr-m14en_nosym` ep4 | 47.2% | 54.4% | +7.2 | **43.2%** | 43 | 79 |
| DG-ICL-Sym | `112859_af_h-cr-m14en_dpi_ha` ep4 | 50.0% | 56.8% | +6.8 | **29.6%** | 32 | 66 |
| DG-ICL-Flip | `003524_af_h-cr-m14en_nosym_swap_per_instance` ep4 | 44.2% | 49.6% | +5.4 | **39.2%** | 45 | 72 |
| **Pre-trained** | no LoRA | **50.0%** | **57.2%** | **+7.2** | **31.8%** | 26 | 62 |

CREMA-D control:

| condition | speech | text | disagree | speech only | text only |
|---|---|---|---|---|---|
| Instruction Tuning | 82.2% | 17.4% | 82.6% | 335 | 11 |
| DG-ICL-Sym | 77.6% | 16.2% | 82.2% | 310 | 3 |
| DG-ICL-Flip | 79.8% | 17.0% | 84.6% | 324 | 10 |
| Pre-trained | 51.6% | 14.4% | 57.6% | 187 | 1 |

### What it shows

1. **The control validates the setup.** On CREMA-D the transcript carries
   nothing: text collapses to 14 to 17 percent against 78 to 87 percent from
   audio, and `speech only correct` beats `text only` by two orders of
   magnitude. The pipeline measures what we intend it to.

2. **[F] Axis B has headroom on both backbones; the kill criterion is cleared.**
   MELD disagreement is 48 to 52 percent on Qwen and 30 to 43 percent on AF3.
   The plan's abort condition was disagreement near zero. E0b is worth building,
   on both backbones.

3. **[F] Every fine-tuned model leans lexical, but the origin differs by
   backbone.** This was initially written up as "fine-tuning creates the lean",
   on the strength of the Qwen base alone. The AF3 base refutes the general
   form, and the corrected claim is narrower:

   | backbone | base gap | fine-tuned gap | reading |
   |---|---|---|---|
   | Qwen2-Audio | **-0.4** | +10.0 to +10.6 | training **induces** the lean |
   | Audio Flamingo 3 | **+7.2** | +5.4 to +7.2 | lean **pre-exists**; training leaves it alone |

   On Qwen the pre-trained model is balanced (29.2 speech vs 28.8 text) and
   training opens a 10-point gap, improving the text pathway by about 23 points
   on held-out MELD against only 11 to 12 for audio. On AF3 the pre-trained
   model already sits at +7.2 and fine-tuning does not widen it.

   Consequence for Chapter 2: an intervention that works by undoing a bias our
   own training introduced would be a Qwen story. To claim generality the method
   has to correct a bias that is already present in the foundation model.

3b. **[F] On held-out MELD, the 3-task mixture bought AF3 nothing.** Base AF3
   speech is 50.0 and DG-ICL-Sym is also 50.0; IT falls to 47.2 and Flip to
   44.2. On in-mixture CREMA-D the same training lifts AF3 from 51.6 to 78-82.
   Large in-domain gains, no held-out transfer, and Flip actively hurts. Chapter 2
   should not assume training will move MELD numbers on this backbone.

4. **[F] The surrogate-label scheme is not the cause.** On Qwen, IT, Sym and
   Flip sit within 0.6 points of each other on the lexical gap (+10.6, +10.4,
   +10.0) and within 4 points on disagreement. The lean is a property of the
   training setup, not of how labels are represented. This supports H2:
   Chapter 1 changed behaviour under label-name conflict and did not touch
   evidence selection.

5. **[F] Magnitude is backbone dependent, so claims must be per backbone.** AF3
   leans lexically too, but less (+5.4 to +7.2 against Qwen's +10.0 to +10.6),
   is much better on audio (44 to 50 percent against 40 to 42), and disagrees
   far less (29.6 to 43.2 percent against 47.6 to 51.6). AF3 fuses audio by
   cross-attention while Qwen concatenates audio tokens, which is a plausible
   mechanism and worth a sentence in the paper. Note also that AF3 conditions
   vary more among themselves than Qwen's do, so the "training setup, not label
   scheme" conclusion in point 4 is cleanly supported on Qwen and only loosely
   on AF3.

   Method note: this is the second claim in this log that a single backbone made
   look general. Run both backbones, and the pre-trained baseline, before
   writing any E0 finding as a headline.

### Caveats to carry forward

- **High disagreement in a weak model is noise, not arbitration.** Pre-trained
  Qwen disagrees on 75.8 percent of MELD items but gets 58 percent of them wrong
  with both modalities. Its two views differ because both are unreliable. Only
  the fine-tuned disagreement figures (~50 percent on Qwen, ~30 to 43 on AF3)
  describe genuine conflict, because accuracy there is much higher.
- **Fine-tuning reduced disagreement** (75.8 to about 50 on Qwen) while making
  one modality dominant. Training made the two views more consistent and more
  skewed at the same time.
- MELD was never in any training mixture, so these are held-out numbers for
  every fine-tuned condition.

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

---

## E0b-1: does the definition decide which evidence the model uses?

**Question.** H3 assumed routing was at least partially present and trainable.
First, is it there at all? Four passes per checkpoint on the same 500 items, all
`speech_only` except the text reference, so the audio carries both channels and
the legend is the only thing that differs between the routed passes.

Launcher `hpc/submit_e0b_routing.sh`, analysis `experiments/e0b_routing.py`.
Qwen2-Audio, 14 Nov 2026.

### Routing gap on MELD (the measurement)

| condition | items disagreeing | route_tone = P_speech | route_words = P_speech | **gap (toward speech)** | gap (toward text) |
|---|---|---|---|---|---|
| Instruction Tuning | 256 (51.2%) | 77.0% | 78.5% | **-1.6** | -0.4 |
| DG-ICL-Sym | 251 (50.2%) | 67.7% | 65.3% | **+2.4** | +0.0 |
| DG-ICL-Flip | 195 (39.0%) | 57.9% | 60.0% | **-2.1** | -3.6 |
| Pre-trained | 390 (78.0%) | 72.3% | 73.8% | **-1.5** | -1.5 |

**[F] The routing gap is zero.** Every condition sits between -3.6 and +2.4
points, with no consistent sign. Telling the model which modality decides does
not move its answer toward that modality.

### The CREMA-D control, which is decisive

CREMA-D transcripts carry no emotion: `text_only` scores 14.8 to 15.0 percent.
So a legend saying "the WORDS express anger, regardless of the tone of voice"
should collapse accuracy toward that floor if the model follows it. Accuracy on
all 500 CREMA-D items:

| condition | speech + orig | text + orig | speech + route_tone | speech + **route_words** |
|---|---|---|---|---|
| DG-ICL-Sym | 86.0% | 14.8% | 84.4% | **84.2%** |
| Instruction Tuning | 87.8% | 15.0% | 88.2% | **88.0%** |

**[F] The model ignores the routing clause completely.** Instructed to decide on
words that demonstrably carry nothing, it keeps reading prosody and loses 0 to 2
points instead of the ~70 it would lose if it complied. The expected effect is
enormous and the observed effect is nil, so this is a much stronger null than
the MELD gap on its own.

### Sanity controls

On the agreement subset, where the two legends have nothing to arbitrate, they
still disagree on 4.5 percent (base), 7.4 (IT), 10.0 (Sym) and 16.7 (Flip) of
items; Flip trips the built-in warning. So the routing clause does perturb the
model, it just does not route. Prompt sensitivity without instruction following.

**[F] Routed legends slightly improve accuracy regardless of direction**, on
MELD: Sym 23.4 to 27.8/28.6, IT 38.8 to 44.0/43.2, base 28.2 to 32.2/29.6. The
gain is the same whichever modality is named, so it is a verbosity or
specificity effect and must be controlled for in later experiments.

### Caveat: these numbers use the normalised MELD template

E0a ran on the old MELD prompt, E0b on the bullet-legend version, so
`speech_only + orig` differs between them. The drop is uneven and informative:
Sym falls 40.4 to 23.4 while IT, Flip and base move only 1 to 4 points.

**[F] DG-ICL-Sym is by far the most legend-format-sensitive condition**, which
is consistent with Chapter 1 Table 5, where Sym lost the most under paraphrase
(0.663 to 0.571). A model trained to read definitions is also the model most
disturbed when they are reworded.

### What this means for Chapter 2

This is the "routing gap near zero" branch of the plan, so the problem statement
is clean and the training result becomes the whole contribution. The negative
result is also stronger than ALME's: they instruct at the system-prompt level
and find poor compliance, while we route per category, inside the definition,
on a model fine-tuned to follow definitions, and get no compliance at all.

Open risk to state plainly: nothing here shows routing is *trainable*. H3 is now
a bet rather than an extrapolation. The cheapest way to de-risk it before
building CD-DPO is a small supervised probe, training on routed legends with
modality-appropriate targets, to see whether the capability can be induced.

---

## Pending

- SALMONN E0a, optional. It is the scale reference rather than a headline model.
- E0b-1 on AF3, to check the null replicates on the second backbone.
- E0b-2, `speech_text`, only needed to separate "cannot be routed" from "could
  not recover the words from audio". The CREMA-D control makes this less urgent,
  since there the words were irrelevant rather than unrecoverable.
- A trainability probe for H3, per the risk noted above.

## Artifacts

| What | Where |
|---|---|
| Prediction dumps | `~/training/e0a/metrics/<date>/<run>/<run>_p.json` |
| Disagreement subsets | `~/training/e0a/disagreement_{sym,it,flip,base,af_sym,af_it,af_flip,af_base}.json` |
| E0b routing detail | `~/training/e0b/routing_<tag>_<dataset>_<stamp>.json` |
| Run logs | `~/training/e0a/logs/`, `~/training/e0b/logs/` |
| Inference logs | `~/training/symbol_training/logs_inference/<date>/` |

Not in git: all of the above live outside the repo. Only the scripts and this
log are versioned.
