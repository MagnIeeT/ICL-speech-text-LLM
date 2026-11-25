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

### E0b-2: the same test with the transcript supplied (`speech_text`)

E0b-1 left one loophole: with audio only, a null could mean "will not route" or
"could not recover the words to route to". E0b-2 hands the model a labelled
transcript alongside the audio, removing the extraction step. Qwen, 15 Nov 2026.

| condition | legend | = P_speech | = P_text | neither | gap (speech) |
|---|---|---|---|---|---|
| DG-ICL-Sym | route_tone | 29.1% | 45.0% | 25.9% | **+2.4** |
| | route_words | 26.7% | 47.4% | 25.9% | |
| Instruction Tuning | route_tone | 41.8% | 37.5% | 20.7% | **-2.0** |
| | route_words | 43.8% | 37.1% | 19.1% | |
| DG-ICL-Flip | route_tone | 29.7% | 49.7% | 20.5% | **-0.5** |
| | route_words | 30.3% | 50.3% | 19.5% | |

**[F] The loophole is closed and the null holds.** Two facts together:

1. The transcript is legible and highly influential. Supplying it shifts answers
   massively toward `P_text`: Sym goes from 17.9 percent matching `P_text` with
   audio only to 45.0 to 47.4 percent with the transcript present, a swing of
   about 28 points. The model plainly reads and uses it.
2. The routing gap stays at zero anyway: +2.4, -2.0, -0.5. Identical to E0b-1.

So the model cannot be excused on extraction grounds. Handed both channels in
legible form, it shifts toward text regardless of which channel the definition
names as authoritative. Routing fails because the definition does not control
evidence selection, not because the evidence was unavailable.

This is also a second, independent measurement of text dominance: with both
modalities present and no routing instruction obeyed, the lexical channel wins.

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

## E1: the naive surrogate mixture

Per-instance regime sampling: with probability `surrogate_mix` an example is
trained under a permuted-real-label legend (DG-ICL-Flip), otherwise under
synthetic symbols (DG-ICL-Sym). Total training examples are unchanged; only the
regime of each example is resampled.

### Run record

| Run | Date | Checkpoint | Data | Notes |
|---|---|---|---|---|
| pilot, 2k | 15 Nov 2026 | `012455_qa_h-cr-m14en_dpi_mix0.5` ep5 | 2000 capped | first signal |
| full | 16 Nov 2026 | `003616_qa_h-cr-m14en_dpi_mix0.5` ep5 | 18,396 (hvb 11,344 + cremad 5,967 + minds14_en 1,085) | 5 epochs, seed 1, mix 0.5, finished 11:29 |

Full-data training validation, best at epoch 5: `original` 0.8343, `fresh`
0.8389 (hvb 0.6498/0.6570, cremad 0.8726/0.8766, minds14_en 0.9804/0.9832).
Unusually, `fresh` finishes above `original`, which is the regime balance doing
what it was meant to.

### Test-set comparison, pilot checkpoints (`speech_only`, `split=test`, 500/dataset)

Qwen2-Audio. Trained tasks are hvb, cremad, minds14_en; the rest are held out.

| | | `original` | | | | `fresh` | |
|dataset|Sym|Flip|Mix|  |Sym|Flip|Mix|
|---|---|---|---|---|---|---|---|
| hvb (T)          | 0.4847 | 0.5433 | **0.6076** | | **0.5745** | 0.4819 | 0.5255 |
| cremad (T)       | **0.8624** | 0.8558 | 0.8438 | | 0.8586 | **0.8746** | 0.8554 |
| minds14_en (T)   | 0.9458 | 0.9638 | **0.9692** | | 0.8828 | **0.9695** | 0.9640 |
| skit_s2i         | **0.8494** | 0.7717 | 0.7934 | | **0.8473** | 0.7929 | 0.7957 |
| ravdess_song     | **0.4865** | 0.4728 | 0.4621 | | **0.6161** | 0.3292 | 0.5078 |
| minds14_fr       | 0.8498 | 0.8703 | **0.8942** | | 0.8630 | 0.8566 | **0.8746** |
| minds14_ko       | 0.4747 | 0.4577 | **0.5437** | | 0.5308 | 0.4467 | **0.5397** |

### [F] The held-out gain is confined to one task family, which is not held out

The training inventory is hvb + cremad + minds14_en, 38 label words in total.
Label-set overlap against each held-out task:

| held-out dataset | labels | shared with training | overlap | new labels |
|---|---|---|---|---|
| minds14_fr   | 14 | 14 | **100%** | none |
| minds14_ko   | 14 | 14 | **100%** | none |
| ravdess_song |  6 |  4 | 67% | `calm`, `fearful` |
| skit_s2i     | 14 |  0 | **0%** | all 14 |

The mixture beats Sym on exactly the two 100%-overlap datasets and loses on
both others, in `original` and in `fresh` alike.

**The first explanation was wrong.** It attributed the gain to extra practice
emitting the 38 training label words, since the flip regime permutes within the
task's own label set. That cannot be right: under `fresh` the targets are
synthetic symbols and the real label words never appear, so word-level practice
cannot produce a `fresh` gain. There is one.

**Label overlap was a proxy for task identity.** `minds14_fr` and `minds14_ko`
have an identical label set, identical per-label definitions and an identical
prompt template to `minds14_en`, which is a *training* task. Only the spoken
language differs. They are the trained task with different audio, not
independent held-out tasks. Regrouping by task family:

| family | dataset | `original` Sym→Mix | `fresh` Sym→Mix |
|---|---|---|---|
| MInDS-14 | minds14_en (T) | 0.9458 → **0.9692** | 0.8828 → **0.9640** |
| | minds14_fr | 0.8498 → **0.8942** | 0.8630 → **0.8746** |
| | minds14_ko | 0.4747 → **0.5437** | 0.5308 → **0.5397** |
| emotion | cremad (T) | 0.8624 → 0.8438 | 0.8586 → 0.8554 |
| | ravdess_song | 0.4865 → 0.4621 | 0.6161 → 0.5078 |
| dialogue act | hvb (T) | 0.4847 → **0.6076** | 0.5745 → 0.5255 |
| banking intent | skit_s2i | 0.8494 → 0.7934 | 0.8473 → 0.7957 |

The mixture improves **all six MInDS-14 cells** and is neutral or harmful
everywhere else except HVB `original`. The fr/ko `fresh` gain is the
minds14_en `fresh` gain (+0.081, the largest single gain in the table)
transferring to the same task.

**Consequence.** The held-out average is not a generalisation measure for any
method that improves MInDS-14, because two of its four datasets are the trained
MInDS-14 task in another language. The genuinely held-out tasks are Skit-S2I
and RAVDESS-Song, and the mixture loses on both, in both modes.

**Untested hypothesis for the MInDS-14 gain.** minds14_en is by far the
smallest training set (1,085 against hvb 11,344 and cremad 5,967), so it is the
most exposed to overfitting a single legend regime, and alternating regimes acts
as augmentation. If that is the mechanism the gain tracks training-set size and
has nothing to do with surface forms.

### The canonical regime was never trained

`models/symbolAdapter/symbol_training.py:164` samples binary: flip regime
(permuted real labels) or symbol regime. Plan Section 7 E1 specifies three
regimes including canonical. Only two were built, so a label word has never once
been paired with its own definition during training, under any condition.

Note that a perfect definition follower should need no canonical practice: under
canonical evaluation the definition is correct and the attached word is correct,
so following the definition suffices. The canonical deficit therefore indicates
one of two things, which predict differently and can be separated from the
existing prediction dumps at no compute cost:

1. **Output-space drift.** Sym targets are always two-token nonsense; canonical
   targets are real words. The failure is mechanical. Predicts canonical errors
   skew out-of-inventory or symbol-shaped.
2. **Learned distrust of the label word.** The word is never informative in
   training, so the model discounts a cue that canonical makes correct. Predicts
   canonical errors are in-inventory but semantically wrong.

The same analysis on `fresh` found errors almost entirely wrong-but-valid. It
has not been run on `original`.

### On broad-vocabulary flip

Drawing the flip regime's words from a broad vocabulary rather than the task's
own label set would remove the inventory-reuse channel. It is a **diagnostic,
not a fix**: in Sym the word is meaningless, in Flip it is another label's word,
in broad-vocab flip it is unrelated, and in none of them does a word ever match
its own definition. It would add a fourth never-matching surface type while
canonical remains the one form never practised. It does separate hypotheses 1
and 2 above, which is its value.

### Full-data mixture result (`sp_mixfull`, 16 Nov 2026)

`003616_qa_h-cr-m14en_dpi_mix0.5` epoch 5, `speech_only`, test split, 500 per
dataset. Sym and Flip are the paper's full-data checkpoints; `mix2k` is the
pilot.

| family | dataset | role | orig Sym | orig Flip | orig Mix2k | orig MixFull | fresh Sym | fresh Flip | fresh Mix2k | fresh MixFull |
|---|---|---|---|---|---|---|---|---|---|---|
| MInDS-14 | minds14_en | T | 0.9458 | 0.9638 | 0.9692 | **0.9700** | 0.8828 | 0.9695 | 0.9640 | **0.9752** |
| | minds14_fr | H | 0.8498 | 0.8703 | **0.8942** | 0.8662 | 0.8630 | 0.8566 | **0.8746** | 0.8474 |
| | minds14_ko | H | 0.4747 | 0.4577 | **0.5437** | 0.4374 | 0.5308 | 0.4467 | **0.5397** | 0.5296 |
| emotion | cremad | T | 0.8624 | 0.8558 | 0.8438 | **0.8768** | 0.8586 | **0.8746** | 0.8554 | 0.8644 |
| | ravdess_song | H | 0.4865 | 0.4728 | 0.4621 | **0.5297** | **0.6161** | 0.3292 | 0.5078 | 0.5760 |
| dialogue act | hvb | T | 0.4847 | 0.5433 | 0.6076 | **0.6599** | 0.5745 | 0.4819 | 0.5255 | **0.6146** |
| banking intent | skit_s2i | H | **0.8494** | 0.7717 | 0.7934 | 0.7564 | **0.8473** | 0.7929 | 0.7957 | 0.8123 |
| | AVG trained | | 0.7643 | 0.7876 | 0.8069 | **0.8356** | 0.7720 | 0.7753 | 0.7817 | **0.8181** |
| | AVG held-out, all | | 0.6651 | 0.6431 | **0.6734** | 0.6474 | **0.7143** | 0.6064 | 0.6795 | 0.6914 |
| | AVG held-out, disjoint | | **0.6680** | 0.6223 | 0.6278 | 0.6431 | **0.7317** | 0.5610 | 0.6518 | 0.6942 |

**[F] The mixture is a trained-task method and does not generalise.** Trained
tasks gain +0.071 canonical and +0.046 symbolic over Sym. Held-out loses in all
four cells: -0.018 and -0.023 on the full held-out set, -0.025 and -0.038 on
the disjoint subset. The pilot's held-out gain did not survive scaling to full
data, and the MInDS-14 family advantage survives only on the trained member.
**This closes E1 as a negative result.**

### E1a: the canonical deficit is not output-space drift

Out-of-inventory predictions under `original`, as a share of errors
(multi-label tasks scored per component):

| model | hvb | cremad | minds14_en | skit_s2i | ravdess_song | minds14_fr | minds14_ko |
|---|---|---|---|---|---|---|---|
| Sym | 15.3% | 0 | 19.0% (4 cases) | 1.3% | 0 | 0 | 0 |
| Flip | 0 | 0 | 0 | 1.8% | 0 | 0 | 0 |
| Mix | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**[F] Hypothesis 1 (output-space drift) is refuted.** The model emits canonical
label words without difficulty and then picks the wrong one. The deficit is
discrimination, not generation. This retires every fix whose rationale was
restoring real-word output: the flip regime's supposed mechanism,
broad-vocabulary flip, and a canonical regime justified on output-space grounds.

**[F] Real label names are a liability on some tasks.** Sym's own correct counts,
canonical against symbolic on the same items: hvb 291/347, cremad 430/428,
minds14_en 341/320, skit_s2i 424/424, ravdess_song 156/176, minds14_fr 91/92,
minds14_ko 61/67. Meaningless symbols beat real label names on three tasks and
tie on three. Within-model, surrogate tuning has not cost canonical ability; the
gap is against IT, which is a different claim from the one Section 3 makes.

### Surface-form invariance (new metric)

Percent of items where the model gives the same answer under canonical and
under novel symbols, mapped back through the legend. Both modes run the same
items in the same order; alignment self-validates, since misalignment makes the
symbol-to-label map inconsistent. All 28 cells came back consistent with
matching `text` fields.

| model | hvb | cremad | minds14_en | skit_s2i | ravdess_song | minds14_fr | minds14_ko | mean |
|---|---|---|---|---|---|---|---|---|
| Sym | 68.4 | 95.6 | 87.6 | 87.0 | 59.1 | 88.7 | 56.3 | 77.5 |
| Flip | 70.8 | 95.4 | 96.7 | 83.6 | 56.8 | 88.7 | 63.9 | 79.4 |
| Mix2k | 77.6 | 92.2 | 97.5 | 88.2 | 56.8 | 92.5 | 73.1 | 82.6 |
| MixFull | 89.4 | 97.6 | 98.3 | 81.0 | 81.8 | 88.7 | 67.2 | **86.3** |

**[F] Invariance dissociates from held-out accuracy.** It rises monotonically
Sym to MixFull, yet MixFull is the most invariant model and the worst mixture on
held-out accuracy. A model can be invariant by collapsing onto one answer and
being consistently wrong. RAVDESS-Song: Sym is right in both modes on 124 items,
MixFull on 156, but counting items right in *at least one* mode, Sym reaches 208
and MixFull 176. **Invariance alone is therefore not a valid primary metric**; it
has to be paired with a capability measure.

### Oracle headroom across surface forms

Items right under a single surface against right under either of two:

| model | dataset | single | either | headroom (pp) |
|---|---|---|---|---|
| Sym | ravdess_song | 156 | 208 | +19.7 |
| Sym | hvb | 282 | 355 | +14.6 |
| Sym | minds14_ko | 61 | 76 | +12.6 |
| Sym | skit_s2i | 424 | 443 | +3.8 |

Two drawings already expose up to 20 points that single-surface inference
discards. The model often knows the answer under one legend and not another, and
which one is arbitrary. This is an upper bound, not an achievable number.

Scripts: `/tmp/.../scratchpad/e1a_errors.py`, `invariance.py` (not durable;
reproduce from the dumps in `~/training/eval_speech/metrics/2026-11-16/`).

### Corrected comparison against the paper (full test sets)

The 16 Nov evaluation capped three datasets at 500 samples while the paper's
`submit_symbol_inference_test.sh` uses `MAX_VAL_SAMPLES=0`, the full test set.
Re-run on 17 Nov as `sp_mixfull_full3` (hvb 6121, cremad 738, skit_s2i 1400).
The correction is small but Skit-S2I is held out, so it mattered in principle:

| dataset | mode | 500-cap | full | delta |
|---|---|---|---|---|
| hvb | original | 0.6599 | 0.6646 | +0.005 |
| hvb | fresh | 0.6146 | 0.6413 | +0.027 |
| cremad | original | 0.8768 | 0.8680 | -0.009 |
| cremad | fresh | 0.8644 | 0.8582 | -0.006 |
| skit_s2i | original | 0.7564 | 0.7515 | -0.005 |
| skit_s2i | fresh | 0.8074 | 0.8074 | 0.000 |

Baselines are the paper's own three-seed means from
`tables/appendix_tables/per_task_qwen.tex` and
`~/training/symbol_training/metrics/PAPER_qwen_test_2026-09-02.csv`. Mix is one
seed. Prior-work rows (Wei Symbol Tuning, Agrawal Flipped-Label) omitted.

| dataset | mode | role | Pre-trained | IT | DG-ICL-Flip | DG-ICL-Sym | Mix |
|---|---|---|---|---|---|---|---|
| HVB | original | T | 0.023 | **0.674** | 0.655 | 0.572 | 0.665 |
| HVB | symbolic | T | 0.030 | 0.314 | 0.594 | 0.636 | **0.641** |
| CREMA-D | original | T | 0.541 | 0.867 | 0.854 | 0.832 | **0.868** |
| CREMA-D | symbolic | T | 0.564 | 0.718 | 0.854 | 0.851 | **0.858** |
| MInDS-14 en | original | T | 0.794 | **0.974** | 0.973 | 0.865 | 0.970 |
| MInDS-14 en | symbolic | T | 0.511 | 0.877 | 0.962 | 0.974 | **0.975** |
| Skit-S2I | original | H | 0.795 | **0.851** | 0.762 | 0.798 | 0.752 |
| Skit-S2I | symbolic | H | 0.744 | **0.810** | 0.803 | 0.767 | 0.807 |
| RAVDESS-Song | original | H | 0.620 | **0.660** | 0.508 | 0.520 | 0.530 |
| RAVDESS-Song | symbolic | H | 0.592 | 0.484 | 0.645 | **0.668** | 0.576 |
| MInDS-14 fr | original | H | 0.611 | 0.848 | **0.878** | 0.765 | 0.866 |
| MInDS-14 fr | symbolic | H | 0.387 | 0.743 | 0.856 | **0.866** | 0.847 |
| MInDS-14 ko | original | H | 0.186 | **0.531** | 0.522 | 0.451 | 0.437 |
| MInDS-14 ko | symbolic | H | 0.146 | 0.443 | 0.454 | **0.521** | 0.530 |
| AVG trained | original | | 0.452 | **0.838** | 0.827 | 0.756 | 0.834 |
| AVG trained | symbolic | | 0.368 | 0.636 | 0.803 | 0.820 | **0.825** |
| AVG held-out | original | | 0.553 | **0.723** | 0.668 | 0.634 | 0.646 |
| AVG held-out | symbolic | | 0.467 | 0.620 | 0.692 | **0.703** | 0.690 |

**[F] The mixture is best-of-both on trained tasks and neither on held-out.**
Trained: 0.834 canonical against IT's 0.838, and 0.825 symbolic, the best
symbolic average of any method. Held-out: 0.646 canonical against IT's 0.723,
and 0.690 symbolic against Sym's 0.703. It beats neither baseline on that
baseline's own strength. This is the result that closes E1.

## E2: surface-form consistency objective

Per-instance, two legends for the same audio in one step, coupled by a
divergence between their distributions over *definitions*:

    L = NLL(answer | x, legend_a) + NLL(answer | x, legend_b)
      + weight * D(p_a, p_b)

`p` is read off the logit row that predicts the first answer token, restricted
to the K candidate surfaces and renormalised. One forward pass per legend, no
candidate-by-candidate scoring, no reference model, no preference pair. Both
passes are in the same step with no optimizer step between them, since `D` is a
function of both outputs at once.

Code: `models/symbolAdapter/consistency_loss.py` (objective, with a
`DIVERGENCES` registry), `symbol_training.py` (`_build_view`,
`_cons_dataset_eligible`, `_consistency_step`), flags
`--consistency_weight|_views|_divergence`.

### Design decisions and why

**Views are `sym,flip`, not `sym,orig`.** HVB takes a single pass through the
ordinary mixture path, so it sees sym-or-flip. Coupling sym with `orig` on the
other two datasets would have split the regime: canonical labels on 38 percent
of the data and `flip` on the other 62, for no principled reason. `sym,flip`
keeps every dataset inside the same two surface families.

It also means no canonical *legend* is ever shown, so canonical evaluation stays
close to an unseen surface type. One caveat: `generate_swap_mapping_for_labels`
uses `random.shuffle`, a permutation rather than a derangement, so some labels
keep their own name as fixed points. Expected count is about 1 in 14 per
instance, observed in the launch log (`abroad`->`abroad`,
`direct_debit`->`direct_debit`). So a small amount of canonical pairing leaks in.
This matches the paper's DG-ICL-Flip, which uses the same function, so it is not
a departure from the baseline, but it does weaken the claim: a canonical gain
under `sym,flip` is *mostly* surface-form generalisation rather than entirely.
Switching to a true derangement would remove the leak and is a one-line change
if the result turns out to hinge on it.

**HVB cannot carry the term**, for two independent reasons: it is multi-label,
so a distribution over K definitions is the wrong object, and its canonical
surfaces collide on first token (`answer_*`, `question_*`, `statement_*` are
three groups, 10 distinct first tokens out of 18). It takes one pass via
`_apply_symbol_replacement`, so its regime follows `surrogate_mix` and matches
how the baselines train it. That also saves roughly 8 hours: 16 rather than 26.

**First-token distinctness is a paper property, not a new constraint.** The
objective reads one logit row, so two surfaces sharing a first token would
occupy one logit and the divergence could not separate those definitions. The
paper already had this: its run log reads `Loaded vocab_h2_scores.json:
hard_pool=336, easy_pool=633 (first-token deduplicated)`, and commit `12b6d34`
is titled "Deduplicate symbol pools by first BPE token to reduce gradient
interference". That machinery was removed in the release cleanup, so the
**mixture** run trained on symbols whose first tokens could collide, a small
undocumented departure from the paper's recipe.
`generate_fresh_symbols(distinct_first_token=True)` restores it, enforced when
the objective is on.

First-token distinctness of canonical label sets, Qwen2-Audio tokenizer:

| dataset | labels | distinct first tokens |
|---|---|---|
| CREMA-D | 6 | 6 |
| MInDS-14 en / fr / ko | 14 | 14 |
| Skit-S2I | 14 | 14 |
| RAVDESS-Song | 6 | 6 |
| HVB | 18 | **10** |

**`js` is the default divergence, not `sym_kl`.** First smoke test with
`sym_kl` at weight 0.5: `D` ran 1.6 to 14.2 against an NLL sum near 1 to 2, so
the divergence was up to 84 percent of the loss. That inverts the intended
balance and points straight at the failure mode already measured, consistency
bought by collapsing onto one confident answer. `js` is bounded by `ln 2`;
observed 0.24 to 0.49, contributing 5 to 8 percent.

### Smoke tests

| run | result |
|---|---|
| `012247 ... consSO0.5` (sym_kl) | ran clean; 286/300 eligible steps got `D`; 14 lost to symbol first-token collisions; `D` too large |
| `013939 ... consSO0.5` (js, fixes) | **300/300 (100%)**, 0 collisions, `Never eligible: ['hvb']`, loss 3.53 to 3.18 over the epoch, 1.7 it/s |

Validation numbers from both are meaningless (one epoch, 150 samples/dataset).

### Full run, launched 17 Nov 01:54

`015436_qa_h-cr-m14en_dpi_mix0.5_consSF0.5`, GPU 1, full data 18,396, 5 epochs,
seed 1, `sym,flip` at weight 0.5 with `js`, `SURROGATE_MIX=0.5`. Roughly 16
hours. Compares against the Mix column above and the paper baselines.

### Early read: epoch 3 checkpoint on held-out test (17 Nov)

> Superseded as a headline by the epoch 5 section below: epoch 3 is not the
> checkpoint any held-out-free criterion selects.

`sp_consSF_ep3_heldout`, `lora_epoch3_periodic.pt`, test split, full test sets
(`MAX_VAL_SAMPLES=0`), `speech_only`, `original,fresh`, GPU 2. Mix uses the
full-test `sp_mixfull_full3` number for Skit-S2I. Paper columns are 3-seed means;
Cons is one seed at a mid-training epoch.

| dataset | mode | IT | DG-ICL-Flip | DG-ICL-Sym | Mix | Cons ep3 |
|---|---|---|---|---|---|---|
| Skit-S2I | original | **0.851** | 0.762 | 0.798 | 0.752 | 0.834 |
| Skit-S2I | symbolic | **0.810** | 0.803 | 0.767 | 0.807 | 0.798 |
| RAVDESS-Song | original | 0.660 | 0.508 | 0.520 | 0.530 | **0.681** |
| RAVDESS-Song | symbolic | 0.484 | 0.645 | **0.668** | 0.576 | 0.655 |
| MInDS-14 fr | original | 0.848 | 0.878 | 0.765 | 0.866 | **0.897** |
| MInDS-14 fr | symbolic | 0.743 | 0.856 | 0.866 | 0.847 | **0.911** |
| MInDS-14 ko | original | **0.531** | 0.522 | 0.451 | 0.437 | 0.513 |
| MInDS-14 ko | symbolic | 0.443 | 0.454 | 0.521 | **0.530** | 0.494 |
| AVG held-out | original | 0.723 | 0.668 | 0.634 | 0.646 | **0.731** |
| AVG held-out | symbolic | 0.620 | 0.692 | 0.703 | 0.690 | **0.715** |

**[F] First method to top both held-out averages at once.** Original 0.731
against IT 0.723, symbolic 0.715 against Sym 0.703. Against Mix (same data, same
regimes, no coupling) it gains +0.085 original and +0.025 symbolic, and wins 6 of
8 cells. The largest gains are on canonical labels: Skit-S2I +0.082, RAVDESS-Song
+0.151, the cells where every surrogate method had lost to IT.

**Consistency, the actual goal:** it is the best method in 3 of 8 cells and never
the worst. Its largest shortfall to the best is 0.036 (MInDS-14 ko symbolic).
Every other method has at least one cell more than 0.08 behind.

**Caveats.** One seed against 3-seed means; the margins over IT and Sym (0.008,
0.012) are inside typical seed spread, so "ties the best on both" is the safe
reading until seeds 2 and 3 run. Epoch 3 was chosen before seeing any held-out
number, but the epoch 5 checkpoint is the protocol one. MInDS-14 ko symbolic has
2 of 119 invalid predictions. Doubled supervision is not yet separated from the
coupling (see Known gap below).

### Protocol checkpoint: epoch 5 on held-out test (17 Nov night)

`sp_consSF_ep5_heldout`, `lora_epoch5_final.pt`, same protocol as above. Epoch 5
is the legitimate pick: `hpc/prune_checkpoints.py` keeps the best epoch by
trained-task validation, and that is epoch 5 in both modes (0.838 avg). It also
deleted the epoch 1 to 4 checkpoints, so epoch 3 can no longer be re-evaluated.

| dataset | mode | IT | DG-ICL-Sym | Mix | Cons ep3 | Cons ep5 |
|---|---|---|---|---|---|---|
| Skit-S2I | original | **0.851** | 0.798 | 0.752 | 0.834 | 0.804 |
| Skit-S2I | symbolic | **0.810** | 0.767 | 0.807 | 0.798 | 0.799 |
| RAVDESS-Song | original | 0.660 | 0.520 | 0.530 | **0.681** | 0.639 |
| RAVDESS-Song | symbolic | 0.484 | **0.668** | 0.576 | 0.655 | 0.614 |
| MInDS-14 fr | original | 0.848 | 0.765 | 0.866 | **0.897** | 0.887 |
| MInDS-14 fr | symbolic | 0.743 | 0.866 | 0.847 | **0.911** | 0.892 |
| MInDS-14 ko | original | **0.531** | 0.451 | 0.437 | 0.513 | 0.382 |
| MInDS-14 ko | symbolic | 0.443 | 0.521 | **0.530** | 0.494 | 0.354 |
| AVG held-out | original | 0.723 | 0.634 | 0.646 | **0.731** | 0.678 |
| AVG held-out | symbolic | 0.620 | **0.703** | 0.690 | 0.715 | 0.665 |

**[F] The epoch 3 result does not survive to the protocol checkpoint.** Held-out
falls by about 0.05 in both modes between epochs 3 and 5 while trained-task
validation rises (0.821 to 0.838). At epoch 5 it still beats Mix on original
(+0.032) but is below IT, Sym and Mix on symbolic. MInDS-14 ko accounts for most
of the drop (-0.13 and -0.14). Skit-S2I symbolic has 53 of 1400 invalid
predictions at epoch 5, against none at epoch 3.

Trained tasks, same checkpoint (`sp_consSF_ep5_trained`, full test sets):

| dataset | mode | IT | DG-ICL-Sym | Mix | Cons ep5 |
|---|---|---|---|---|---|
| HVB | original | **0.674** | 0.572 | 0.665 | 0.651 |
| HVB | symbolic | 0.314 | 0.636 | 0.641 | **0.656** |
| CREMA-D | original | 0.867 | 0.832 | **0.868** | 0.867 |
| CREMA-D | symbolic | 0.718 | 0.851 | 0.858 | **0.866** |
| MInDS-14 en | original | **0.974** | 0.865 | 0.970 | 0.967 |
| MInDS-14 en | symbolic | 0.877 | 0.974 | **0.975** | **0.975** |
| AVG trained | original | **0.838** | 0.756 | 0.834 | 0.828 |
| AVG trained | symbolic | 0.636 | 0.820 | 0.825 | **0.832** |

On trained tasks it keeps Mix's best-of-both profile (symbolic best of any
method, original within 0.010 of IT). The cost of the objective is entirely on
held-out.

**Reading.** The coupling helps held-out early and then the model over-fits the
three training tasks. The epoch 3 numbers are not a legitimate headline: no
held-out-free criterion would have picked epoch 3. Two consequences:
1. Epoch selection needs a held-out validation split that is not a test set (E1c),
   or a fixed shorter schedule decided in advance and applied to every method.
2. Future runs must keep every epoch's checkpoint (the pruner removed them here).

### Three follow-up runs, launched 17 Nov 23:38

All: full data, 5 epochs, seed 1, `SURROGATE_MIX=0.5`, `js`, val 500,
`KEEP_ALL_CHECKPOINTS=true` (every epoch kept for diagnostics; the reported
checkpoint is still chosen by trained-task validation, as for the paper).

| GPU | run | what it isolates |
|---|---|---|
| 1 | `233819_..._consSF0.5_shuf` | sym,flip at 0.5 with independent legend order per view: name and position both vary |
| 6 | `233824_..._consSF0` | control: same two views and passes at weight 0; D logged, not trained |
| 7 | `233829_..._consSsame0.5_shuf` | sym,same at 0.5, shuffled: identical symbols, only line order differs |

Verified from the batch-0 prompts: in the `same` run both views carry the same
symbol-to-definition bindings in a different order; in the shuffle run the flip
view has both different words and a different order. Control total equals
`nll_a + nll_b`.

### Epoch 3 held-out, three follow-ups (18 Nov, diagnostic)

Epoch 3 periodic checkpoints, held-out test, full sets, same protocol. Diagnostic
only: the paper rule picks the epoch by trained-task validation. Comparing the
variants with each other at a fixed epoch is fair.

| dataset | mode | IT | Sym | Mix | Cons | Shuffle | Control |
|---|---|---|---|---|---|---|---|
| Skit-S2I | original | **0.851** | 0.798 | 0.752 | 0.834 | 0.771 | 0.812 |
| Skit-S2I | symbolic | 0.810 | 0.767 | 0.807 | 0.798 | 0.726 | **0.830** |
| RAVDESS-Song | original | 0.660 | 0.520 | 0.530 | 0.681 | 0.667 | **0.696** |
| RAVDESS-Song | symbolic | 0.484 | **0.668** | 0.576 | 0.655 | 0.631 | 0.663 |
| MInDS-14 fr | original | 0.848 | 0.765 | 0.866 | **0.897** | 0.834 | 0.851 |
| MInDS-14 fr | symbolic | 0.743 | 0.866 | 0.847 | **0.911** | 0.810 | 0.896 |
| MInDS-14 ko | original | **0.531** | 0.451 | 0.437 | 0.513 | 0.466 | 0.515 |
| MInDS-14 ko | symbolic | 0.443 | 0.521 | 0.530 | 0.494 | 0.368 | **0.544** |
| AVG held-out | original | 0.723 | 0.634 | 0.646 | **0.731** | 0.685 | 0.719 |
| AVG held-out | symbolic | 0.620 | 0.703 | 0.690 | 0.714 | 0.634 | **0.733** |

**[F] The control matches the consistency run, so the coupling is not what
helps.** Weight 0 with the same two views is within 0.012 of Cons on original and
0.019 above it on symbolic. The gain over Mix (+0.07 / +0.04) comes from training
every instance under both surrogate regimes in the same step, not from the
divergence term. One seed, so the Cons-vs-Control ordering itself is noise-level;
the point is that removing the coupling does not remove the gain.

**[F] Shuffling the legend hurts at epoch 3** (-0.046 original, -0.080
symbolic against Cons). Plausible cause: training sees random order, evaluation
sees the fixed paper order, a train/test mismatch the unshuffled runs do not have.

**Same + shuffle (sym,same, epoch 3):** Skit-S2I 0.838 / 0.797, RAVDESS-Song
0.763 / 0.671, MInDS-14 fr 0.720 / 0.781, MInDS-14 ko 0.403 / 0.385; AVG 0.681
original, 0.659 symbolic. Best of any method on RAVDESS-Song (original +0.067 over
the next), but MInDS-14 fr/ko collapse (-0.13 to -0.16 against Control). Its
two-view steps contain symbols only, never permuted real words, so this fits the
earlier finding that the MInDS-14 fr/ko gains come from training on real label
words for the same intent inventory (they are the trained MInDS-14 en task in
another language). Position variation alone does not carry the gain.

### Known gap: no A-only control

`--consistency_weight 0` does **not** give a two-view-without-coupling control;
the orchestrator gates the whole two-view path on the weight being positive, so
weight 0 falls back to the single-legend mixture. The full run therefore changes
two things at once on the eligible datasets: both legends get NLL supervision
(doubled passes) and the divergence couples them. A gain cannot yet be
attributed to the coupling alone. Fix is a `--consistency_control` flag, a few
lines, not yet written.

---

## Pending

- **[DONE 17 Nov: `CONS-RUN` line every 200 batches]** **Periodic NLL logging in the consistency loop.** The `CONSISTENCY` line is
  gated on `batch_idx < 2`, so only two steps per epoch report the `nll_a`/
  `nll_b` split. Add a running mean of `nll_a`, `nll_b` and `D` every ~200
  steps. Without it we cannot see whether the flip side catches up to the symbol
  side, which the launch log suggests is the interesting asymmetry
  (`nll_sym=0.003` against `nll_flip=2.96` at step 0).
- **[DONE 17 Nov, smoke-tested]** **`--consistency_control` flag.** Two views at weight 0, to separate the
  coupling from the doubled supervision. See "Known gap" under E2.
- **True derangement for the flip view**, if a canonical gain turns out to hinge
  on the permutation fixed points. One-line change.

- SALMONN E0a, optional. It is the scale reference rather than a headline model.
- E0b-1 on AF3, to check the null replicates on the second backbone.
- E0b-2, `speech_text`, only needed to separate "cannot be routed" from "could
  not recover the words from audio". The CREMA-D control makes this less urgent,
  since there the words were irrelevant rather than unrecoverable.
- A trainability probe for H3, per the risk noted above.

## Gold-definition diagnostic, Qwen held-out test (19 Nov)

**Question.** Before training a conditional (mDPO-style) loss, does the model still
pick the gold label once only that label's definition is removed? If it does, it is
answering from something other than the definition, and the loss has a gap to close.

**Setup.** Held-out test (Skit, RAVDESS-Song, MInDS fr/ko), full sets, `speech_only`,
one seed per method: IT `004932` ep3, Sym `003002` ep4, Flip `014804` ep4, Mix
`003616` ep5. New eval flag `STRIP_GOLD_DEFINITION=true` turns `- label: definition`
into `- label` for the gold class only; every other definition stays. Flipped mode
now uses a sha256-seeded derangement, so every model gets the same swap (the old
`hash()` seed differed per process). Accuracy, not macro-F1. Per-utterance pairs use
an order-preserving alignment because some batches in the first IT/Mix/Sym runs were
skipped (CUDA OOM, two evals per GPU).

| (all 4 tasks) | IT | Sym | Flip | Mix |
|---|---|---|---|---|
| Original, full definitions | 0.818 | 0.754 | 0.717 | 0.732 |
| Original, gold definition stripped | 0.701 | 0.693 | **0.361** | 0.516 |
| ... right answers kept | 85% | **89%** | 49% | 68% |
| Counterfactual swap | 0.233 | 0.677 | **0.767** | 0.745 |
| ... right answers kept | 25% | 77% | **95%** | 94% |
| ... answers giving the old prior word | **60%** | 15% | 3% | 7% |
| Symbols, full definitions | 0.749 | 0.787 | 0.759 | 0.776 |
| Symbols, gold definition stripped | **0.124** | 0.011 | 0.011 | 0.059 |

Chance is about 0.07 (1/14; 1/6 on RAVDESS). Per-task numbers:
`docs/results/qwen_gold_strip_diag.csv`, script `docs/results/qwen_gold_strip_diag.py`.

**Findings.**
- **Symbols: no gap.** With the gold definition gone, Sym and Flip pick the bare
  gold symbol 1% of the time, below chance. A conditional loss on symbols has
  nothing to push on. Only IT is above chance (0.12, and 0.49 on RAVDESS).
- **Real words: large gap, but not clearly a failure.** Models still pick the bare
  gold word often. Sym keeps 89% of its right answers, more than IT (85%): it
  never trained on real words, so on original labels it reads the name. Flip
  relies on definitions most (49%). A label name like `loan_query` is itself
  evidence, so suppressing it may cost original-label accuracy, where we already
  trail IT.
- **Counterfactual retention replicates AF3.** Flip and Mix keep 94-95% of their
  right answers under the swap; IT keeps 25% and answers the old word 60% of the
  time. On AF3 (saved predictions, 3 seeds) the same pattern holds: Flip keeps
  96%, IT 19%. AF3 Flip's weakness on symbols is invalid output (26% of answers
  are not a legend symbol), not misreading: its accuracy on valid outputs equals
  Sym's (0.85 vs 0.83).

### Each model reads the task best in the label format it was trained on (21 Nov)

Same checkpoints and evals as above, held-out test, accuracy.

| (all 4 tasks) | Original names | Fresh symbols | Swapped names |
|---|---|---|---|
| IT (trained on original) | **0.818** | 0.749 | 0.233 |
| Sym (trained on symbols) | 0.754 | **0.787** | 0.677 |
| Flip (trained on swapped) | 0.717 | 0.759 | **0.767** |

**[F] Sym's loss on original labels is not forgotten label words.** Only 1.3% of
its original-label outputs are invalid (MInDS fr/ko: `cardIssues`, `cashDeposit`,
a symbol-style rewrite of `card_issues`, `cash_deposit`). With only the name left
(gold definition stripped) Sym and IT are equal (0.693 vs 0.701), so name
knowledge is intact. The gap is wrong but valid choices.

**[F] The gap sits in a few confusable label pairs, and each model resolves them
only in its own training format.** Predictions on the 100 true `branch_address`
Skit utterances (definitions: "asks where a bank branch is located" vs
`ifsc_code` "asks for the IFSC code of a branch"):

| Model | Original names | Fresh symbols |
|---|---|---|
| IT | 40% (52 to `ifsc_code`) | 4% (95 to `ifsc_code`) |
| Sym | 2% (96 to `ifsc_code`) | 74% |
| Flip | 14% | 55% |

On the same utterances, Sym is right with symbols and wrong with names in 159
cases, and the reverse in 97; for IT it is 42 and 171. RAVDESS-Song shows a
related pattern on original labels: IT never outputs `calm` (says `sad` 94 times
for 48 sad + 48 calm clips), Sym never outputs `sad` (106 `fearful`), Flip says
`sad` twice. One hypothesis, untested: label-word habits from the training tasks
(CREMA-D has `sad`, no `calm`).

**Reading.** The decision depends on the label format, not only on the audio and
the definitions. This motivates the chapter 2 goal: a speech decision model that
picks among user-given choices from their definitions, giving the same answer
whatever the labels are called (proper names, `label1`, symbols, conflicting
names). Related: E1 oracle headroom (up to 20 points known under one legend but
not another). Caution: E2's two-view consistency loss already tried a version of
this, and its gain came from the two views, not the coupling, and faded by the
protocol epoch.

Scripts (scratchpad, not durable): `per_label.py`, `invalid.py`,
`orig_vs_sym.py`, all reuse `docs/results/qwen_gold_strip_diag.py`.

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
