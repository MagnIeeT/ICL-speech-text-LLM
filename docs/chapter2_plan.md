# Chapter 2 Plan: Definition-Governed Evidence Selection

Status: draft for discussion, 14 Nov 2026. Supersedes the Chapter 2 section of
`thesis_chapters_plan.pdf`. Work happens on the `chapter-2` branch, never on `main`.

---

## 1. One-sentence claim

A category definition should determine not only what a class *means* but which
*evidence* decides it, and preference training on evidence-routed negatives makes
speech-text LLMs honor that routing without the canonical-accuracy loss that
DG-ICL-Sym paid.

## 2. Why this follows from Chapter 1

Chapter 1 established that definitions alone can drive zero-shot SLU. Its own
tables then expose two unfinished pieces:

1. **A trade-off.** DG-ICL-Sym buys symbolic generalization with canonical
   accuracy: 0.836 (IT) to 0.762 (Sym) on AF3 trained tasks, 0.723 to 0.633 on
   Qwen held-out canonical. Cross-entropy raises the probability of the correct
   surrogate but applies no gradient against the competing shortcut.
2. **A second, unmeasured conflict.** Limitation 3 observes that transcripts
   discard affective information. That is the symptom of a deeper issue: when
   the words and the prosody disagree, which one wins, and can the prompt
   control it?

Chapter 1 measured conflict between a *label name* and its definition. Chapter 2
adds conflict between *modalities*, and proposes one training method for both.

## 3. Two conflict axes, one failure mode

| Axis | What conflicts | Ch1 status |
|---|---|---|
| **A: lexical prior** | the surface label's pre-trained meaning vs its assigned definition | measured; Sym retains 68%, IT 21% |
| **B: evidence routing** | what the words say vs what the prosody says | unmeasured |

Both are the same failure: the model falls back on something other than the
definition it was given.

**Positioning against the literature.** Preference optimization for
context-faithfulness already exists in text ([Context-DPO][ctxdpo], [KnowPO][knowpo],
[RPO][rpo]), so the objective is not the contribution. Audio-LLM text dominance is
documented but only diagnostically: [ALME][alme] (arXiv 2602.11488, Mar 2026)
contributes 57,602 audio-text conflict stimuli across eight languages and the Text
Dominance Ratio, shows GPT-4o and Gemini 2.0 Flash follow conflicting text 10 to 26x
more than baseline even when instructed to trust the audio, and **explicitly proposes
no training fix** (verified: its only fine-tuning is a diagnostic ablation). Related
work is circuit analysis ([text dominance circuits][circuits]) or decoding-time only
([context-aware decoding for spoken dialogue][cad]). The gap is a *training* method,
and routing evidence through per-category definitions rather than a global
system-prompt instruction.

## 4. Hypotheses

- **H1 (recover the trade-off).** An explicit negative gradient on the
  prior-implied label recovers canonical accuracy lost by DG-ICL-Sym while
  keeping symbolic generalization. Target: Pareto, no canonical regression
  within noise, gains on symbolic and counterfactual.
- **H2 (two capabilities, not one).** Definition adherence under axis A and
  under axis B are distinct. Chapter 1 improved A only; DG-ICL-Flip and
  DG-ICL-Sym show no reliable routing gap on B.
- **H3 (routing is trainable).** Preference training with evidence-routed
  negatives produces a measurable routing gap on axis B, where instruction-level
  prompting alone fails.
- **H4 (phrasing robustness).** Training across definition phrasings closes the
  paraphrase gap (0.663 to 0.571 in Ch1 Table 5), measured on **held-out**
  phrasings, not the four already in that table.

## 5. Method: evidence-routed preference optimization

For every training instance build a preference pair (y_w, y_l), automatically,
no annotation. The winner is what the active definition points to; the loser is
the fallback the model is tempted by:

| Axis | y_w | y_l |
|---|---|---|
| A | surrogate attached to the correct definition | the canonical word whose pre-trained meaning matches the audio |
| B | label per the modality the definition names authoritative | label per the other modality |

Objective: DPO with an NLL anchor (Formulation 1 of the original draft). Chosen
because `symbol_training_dpo.py` already implements it, the reference model is
free (`disable_adapter_layers()` on the same base, no second model in memory),
and the NLL term guards against DPO depressing the chosen sequence.

ORPO is a footnote if DPO proves unstable. GRPO is not planned: two-token outputs
make rollouts cheap but also low-entropy, risking group collapse. **The optimizer
is an implementation choice, not a contribution, and the paper should say so.**

## 6. Stimulus construction for axis B

CREMA-D and RAVDESS-Song are lexically *uninformative*, not *conflicting*: their
sentences are emotionally neutral by design ("It's eleven o'clock"), so there is
nothing for the prosody to disagree with. They are the control, not the test.

Preferred source is **natural** conflict (MELD, conversational speech with its own
true transcript), because it avoids the artificiality of a transcript that does not
match the audio. Constructed conflict is the fallback:

- **Audio** from CREMA-D, carrying the prosodic (acted) emotion label.
- **Transcript** injected from MELD, carrying its own lexical emotion label.
- Pair them so the two labels disagree. Both ground truths are exact, so both
  routings are scorable.

**Routed legends.** Same audio, same transcript, same symbols, only the routing
clause changes:

```
zeph: utterances where the SPEAKER'S TONE conveys anger, regardless of the words spoken
qolt: utterances where the WORDS express anger, regardless of the tone of voice
```

Four prompt conditions per item:

| Condition | Legend | Purpose |
|---|---|---|
| `agree` | neutral | ceiling; can the model do the task at all |
| `neutral` | no routing clause | default modality bias, our TDR analogue |
| `route_tone` | tone authoritative | routing compliance toward audio |
| `route_words` | words authoritative | routing compliance toward text |
| `trap` | routing clause present, modalities agree | answer must not change |

Guards against the legend acting as a keyword trigger rather than real routing:
symbol labels so the name carries nothing, several paraphrases of the routing
clause, and the `trap` condition.

## 7. Metrics

- **Routing gap**, the headline:
  `gap = P(ans = tone-label | route_tone) - P(ans = tone-label | route_words)`.
  Paired on identical inputs, so it controls for task difficulty and audio quality.
- **Default bias**, from `neutral`: fraction following text when the modalities
  disagree. Comparable in spirit to ALME's TDR.
- **Ceiling**, from `agree`: macro-F1 when nothing conflicts. The routing gap is
  read against this, since a model that cannot hear prosody cannot route to it.
- Chapter 1 metrics unchanged for H1: canonical, symbolic, counterfactual
  macro-F1, `macro_f1_with_invalid`.
- Three seeds, Welch t-test with Hedges correction, as in Chapter 1.

## 8. Experiment ladder

**E0, pilot. Inference only, no training. Run this before anything else.**
Split into two steps so that step 1 needs no new code at all.

**E0a, disagreement mining (zero new code).** For each Chapter 1 checkpoint, run the
same evaluation items twice with the existing flags, `--input_mode speech_only` and
`--input_mode text_only`, on MELD (natural conversational speech where words and
prosody genuinely diverge) and CREMA-D (neutral sentences, so the text carries almost
no emotion signal; this is the control). Then compute, per item, whether the two
unimodal predictions agree.

Outputs:
- **disagreement rate** per backbone and condition: how often the two modalities lead
  the model to different answers. This alone is a reportable number and is the
  quantitative version of Chapter 1's limitation 3.
- **the disagreement subset**, the items where P_speech != P_text, which becomes the
  evaluation set for E0b. Routing is only meaningful where the modalities actually
  conflict for that model.

**E0b, routing probe (needs the `speech_text` mode and routed legends).** On the
disagreement subset, present audio and transcript together and vary only the legend:

| Condition | Legend | Measures |
|---|---|---|
| `neutral` | no routing clause | default modality bias, our TDR analogue |
| `route_tone` | tone authoritative | compliance toward audio |
| `route_words` | words authoritative | compliance toward text |
| `trap` | routing clause present, modalities agree | must not change the answer |
| `instr_only` | routing stated in the system instruction, not per category | ALME-style baseline our training must beat |

Headline measure, computed against the model's own unimodal predictions rather than
an external ground truth, so no new annotation is needed:

```
routing gap = P(answer = P_speech | route_tone) - P(answer = P_speech | route_words)
```

A gap near zero means the definition does not control evidence selection, which is
ALME's finding restated inside our framework. The `instr_only` condition separates
"the model cannot be routed at all" from "the model can be routed, but only at the
category level, not by a global instruction".

The checkpoint audit confirms all four Chapter 1 conditions have loadable weights on
AF3, Qwen2-Audio and SALMONN, so E0 runs entirely on existing weights:

| | IT | DG-ICL-Sym | DG-ICL-Flip | SymbolTuning | FlippedLabel |
|---|---|---|---|---|---|
| AF3 | 25 runs | 43 | 6 | 6 | 3 |
| Qwen2-Audio | 5 | 3 | 3 | 3 | 3 |
| SALMONN | 4 | 8 | 1 | 1 | 1 |

Every outcome is informative, which is why it is the right first move:

| Pilot outcome | What Chapter 2 becomes |
|---|---|
| routing gap ~0 everywhere | clean problem statement; the training result is the whole contribution |
| DG-ICL-Flip already shows a gap | strong Ch1-to-Ch2 bridge; chapter shifts to hardening and generalization |
| disagreement rate near zero | the modalities never conflict for these models; axis B drops to a section |
| `trap` condition shifts answers | the legend is a keyword trigger, not real routing; report as a negative finding |

**E0c, optional, only if E0b is ambiguous.** Constructed conflict in the ALME style:
CREMA-D audio (prosodic label known) with an injected MELD transcript (lexical label
known), giving exact dual ground truth at the cost of an artificial prompt. Keep this
in reserve rather than building it first.

**E1.** Axis-A preference training on real labels only. Tests whether penalizing
the lexical shortcut recovers canonical accuracy without synthetic symbols. (H1)

**E2.** Objective check: SFT on identical pairs vs DPO on identical pairs. This
must use the *same pair data*, not the Chapter 1 Flip run, or it confounds data
with objective.

**E3.** Hybrid: axis-A plus axis-B routed negatives in one checkpoint. The
scientific core. (H2, H3)

**E4.** Multi-view phrasing, trained on a phrasing pool, evaluated on held-out
phrasings generated separately. (H4)

**E5.** External validity: natural disagreement on MELD audio, where text and
prosody genuinely diverge, rather than constructed pairs.

**Side ablation, not a chapter.** Symbol token properties: length 1 vs 2 vs 3,
and leading-BPE-token collisions. Measured on the Qwen tokenizer, uniform
sampling yields a first-token collision in 28% of 18-label HVB draws and 16% of
14-label MInDS-14 draws, which the Chapter 1 hard pool avoided by construction.
One run per setting, fills an ablation table.

## 9. Baselines

Pre-trained, Instruction Tuning, DG-ICL-Sym, DG-ICL-Flip (all from Ch1, weights
exist), Symbol Tuning (Wei), Flipped-Label Tuning (Agrawal), and SFT-on-identical-pairs
from E2. Prompt-only routing (instruction says "trust the audio") is the ALME-style
baseline that our training is meant to beat.

## 10. Code changes required

Nothing here touches `main`.

**For E0 (pilot), in rough order:**

1. **Conflict dataset builder.** New `utils/build_conflict_set.py`: pairs CREMA-D
   audio with MELD transcripts under a disagreement constraint; emits fields
   `audio`, `text` (injected), `tone_label`, `word_label`.
2. **New `input_mode`.** Add `speech_text` alongside `speech_only` and
   `text_only`, so the prompt renders the audio *and* a transcript line. Touches
   `training_configs.py` (choices), `multi_task_dataset.py:364`
   (`_get_main_audio` gates on `"speech" in input_mode`, already compatible), and
   `format_prompt` in all three processors.
3. **Routed legends via existing machinery.** No new flag needed: add
   `route_tone`, `route_words`, `route_none` as entries in the dataset config's
   `prompt_variants`, selected by the existing `--definition_variant`.
4. **Two dataset configs** (`cremad_conflict_tone`, `cremad_conflict_words`)
   differing only in `completion_key`, so the existing evaluation path scores
   against the right ground truth without touching `evaluation_utils.py`.
5. **Metrics.** Routing gap and default bias are computed post hoc from the
   per-condition prediction dumps; a small analysis script, not pipeline code.

**For E1 to E4, later:**

6. Rewrite pair construction in `symbol_training_dpo.py`. `_sample_rejected_symbol()`
   currently samples a random *other symbol*; it must produce the lexical-prior
   foil (axis A) and the other-modality label (axis B). The loss, merged batch,
   and reference-via-adapter-disable scaffolding are reusable as is.
7. Per-instance phrasing sampling for E4: `--definition_variant` currently fixes
   one phrasing per run; E4 needs a per-instance draw from a pool.

**Cleanup before starting:** rename `chapter-2-dspo` to `chapter-2`, and decide
whether to delete the D-SPO modules there too. They are preserved at
`v1.0-paper-submission` either way, and nothing in this plan uses the router,
slot or Gumbel machinery. `symbol_training_dpo.py` must be kept.

## 11. Compute

One RTX A6000. E0 is inference only: 4 conditions x 2 backbones x 5 prompt
conditions x 1 seed, roughly 20 minutes per run after a ~6 minute model load, so
an overnight job. E1 to E4 are LoRA runs on the same three fine-tuning datasets,
comparable to Chapter 1, roughly 2 to 3 days per stage for AF3 plus Qwen at three
seeds. SALMONN stays a single-seed scale reference.

## 12. Risks and kill criteria

| Risk | Signal | Response |
|---|---|---|
| Prosody not perceivable enough for routing to matter | low `agree` ceiling in E0 | drop axis B to a short section, make H1 and H4 the chapter |
| Routing clause acts as keyword trigger, not real routing | `trap` condition changes the answer | report as a negative finding; it is itself interesting |
| Constructed conflict judged artificial | reviewer risk | E5 on naturally disagreeing MELD audio |
| DPO instability, chosen logprob collapse | training curves | NLL anchor already in objective; ORPO as fallback |
| Novelty judged thin vs Context-DPO | reviewer risk | foreground evidence routing and the pair construction, never the optimizer |
| Ch1 numbers shift in review | revision request | do not hard-code Ch1 numbers into Ch2 drafts |

## 13. Dropped from the original draft, with reasons

- **RL or learned search over symbol choice.** Verbalizer-search literature
  reports modest gains at cost scaling with class count; RLPrompt shows the
  optimum is gibberish with no semantic structure; our symbols are already
  meaningless by construction. Internally falsified twice: the easy/hard H2
  ablation showed no effect and D-SPO never became a chapter. Low ceiling.
- **CD-DPO vs CD-ORPO as a headline.** Demoted to an implementation note.
- **"Beat everything" target table.** Replaced by Pareto claims. The old table
  asked the method to exceed IT on canonical *and* exceed DG-ICL-Sym on symbolic
  on every row, which turns a good result into a reported miss.
- **Stage 4 as drafted.** Training on Original/Paraphrase/Terse/Verbose and then
  reporting on Ch1 Table 5, which is those same four phrasings, is train/test
  overlap on the phrasing axis. Held-out phrasings required.

## 14. Open questions

1. Is MELD the right transcript source for injected conflict, or is a text-only
   emotion corpus (GoEmotions, DailyDialog) cleaner? MELD has the advantage of
   being conversational and already configured.
2. Does axis B extend past emotion to intent, or is prosody-vs-words inherently
   a paralinguistic phenomenon? If it is emotion-only, the chapter narrows.
3. Target venue: the original draft said Computer Speech & Language, Speech
   Communication or TASLP. Does the audio-LLM conflict framing push it toward
   TASLP?
4. Sequencing: Chapter 3 (open-world, abstention, hierarchy) carries more novelty
   but more risk. Chapter 2 is faster and closes the weakness a Chapter 1 reviewer
   is most likely to name. Confirm Chapter 2 goes first.

---

## 15. References

[ctxdpo]: https://openreview.net/pdf/2da95227c6c14c9a45c6a4a068648aca1bc93c37.pdf
[knowpo]: https://ojs.aaai.org/index.php/AAAI/article/view/34783/36938
[rpo]: https://arxiv.org/pdf/2501.13726
[alme]: https://arxiv.org/abs/2602.11488
[circuits]: https://www.alphaxiv.org/abs/2610.25599
[cad]: https://arxiv.org/pdf/2606.16472

**Context faithfulness and knowledge conflict (text)**
- Context-DPO: Aligning Language Models for Context-Faithfulness. https://openreview.net/pdf/2da95227c6c14c9a45c6a4a068648aca1bc93c37.pdf
- KnowPO: Knowledge-Aware Preference Optimization for Controllable Knowledge Selection. https://ojs.aaai.org/index.php/AAAI/article/view/34783/36938
- RPO: Retrieval Preference Optimization for Robust RAG. https://arxiv.org/pdf/2501.13726
- Context-faithful Prompting for Large Language Models. https://arxiv.org/pdf/2303.11315

**Audio-LLM modality conflict and text dominance**
- When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration (ALME, TDR). https://arxiv.org/abs/2602.11488
- Text Dominance in Paralinguistic Conflict: Tracing Suppression Through Audio LLM Circuits. https://www.alphaxiv.org/abs/2610.25599
- From Awareness to Adherence: Bridging the Context Gap in Spoken Dialogue Systems via Context-Aware Decoding. https://arxiv.org/pdf/2606.16472
- Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance. https://www.researchgate.net/publication/403069024
- A Survey of Large Audio Language Models: Generalization, Trustworthiness, and Outlook. https://arxiv.org/html/2605.20266v1

**RL and preference optimization for speech**
- Advancing Speech Understanding in Speech-Aware Language Models with GRPO. https://arxiv.org/html/2509.16990
- Group Relative Policy Optimization for Speech Recognition. https://arxiv.org/pdf/2509.01939
- GRPO Improves Multilingual Speech Recognition in Low-Resource Languages. https://arxiv.org/abs/2610.30663
- EMO-RL: Emotion-Rule-Based Reinforcement Learning for Speech Emotion Recognition. https://aclanthology.org/2025.findings-emnlp.1018.pdf
- Enhancing Speech LLMs through Reinforced Behavior Alignment. https://arxiv.org/pdf/2509.03526

**Symbols, verbalizers and prompt optimization (why we are not doing RL over symbols)**
- Symbol tuning improves in-context learning in language models (Wei et al.). https://arxiv.org/pdf/2305.08298
- RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning. https://aclanthology.org/2022.emnlp-main.222.pdf
- Label-Aware Automatic Verbalizer for Few-Shot Text Classification. https://arxiv.org/pdf/2310.12778
- Making Pre-trained Language Models Better Few-shot Learners (LM-BFF, automatic label search). https://arxiv.org/pdf/2012.15723
- Boosting Prompt-Based Self-Training With Mapping-Free Automatic Verbalizer. https://arxiv.org/pdf/2312.04982

**Our own prior work**
- Chapter 1: Definition-Guided In-Context Learning for Zero-Shot Spoken Language Understanding (under review).
- Spoken Language Understanding on Unseen Tasks With In-Context Learning. https://arxiv.org/html/2505.07731
