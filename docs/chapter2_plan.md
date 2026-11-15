# Chapter 2 Plan: Removing the Canonical Cost of Surrogate Tuning

Status: rewritten 15 Nov 2026 after the E0 pilot. Supersedes both the Chapter 2
section of `thesis_chapters_plan.pdf` and the first version of this file, which
was built around evidence routing. Results log: [`docs/e0_results.md`](e0_results.md).
Work happens on `chapter-2`, never on `main`.

---

## 1. The claim

DG-ICL-Sym buys symbolic generalisation by degrading canonical-label
discrimination. That cost is not inherent: it comes from never seeing real
label words as training targets. A training scheme that keeps the surrogate
mechanism while preserving canonical discrimination should match Instruction
Tuning on canonical labels and DG-ICL-Sym on symbolic ones, in one checkpoint.

## 2. Why this and not evidence routing

The first version of this plan proposed making definitions govern which
modality decides a category. E0b killed it. The routing gap is zero on all four
Qwen conditions (-3.6 to +2.4 points, no consistent sign), and the CREMA-D
control is decisive: told to decide on words that score 14.8 percent alone, the
model held 84 to 88 percent, losing 0 to 2 points instead of about 70. There is
no partial capability to amplify, so H3 would have been betting a chapter on
inducing one from zero.

The canonical gap is the opposite situation: documented in Chapter 1's own
tables, localised to identifiable tasks, mechanistically explicable, and with
existence proofs on both sides.

## 3. The problem, with evidence

Qwen2-Audio, canonical (Original) macro-F1 from Chapter 1 Table
`per_task_qwen`:

| task | IT | DG-ICL-Sym | gap | DG-ICL-Flip | gap |
|---|---|---|---|---|---|
| HVB (18-way) | 0.674 | 0.572 | **-0.102** | 0.655 | -0.019 |
| CREMA-D | 0.867 | 0.832 | -0.035 | 0.854 | -0.013 |
| MInDS-14 en | 0.974 | 0.865 | **-0.109** | 0.973 | -0.001 |
| Skit-S2I | 0.851 | 0.798 | -0.053 | 0.762 | -0.089 |
| RAVDESS-Song | 0.660 | 0.520 | **-0.140** | 0.508 | -0.152 |
| MInDS-14 fr | 0.848 | 0.765 | -0.083 | 0.878 | +0.030 |
| MInDS-14 ko | 0.531 | 0.451 | -0.080 | 0.522 | -0.009 |

Aggregate (symbolic / original):

| | Qwen trained | Qwen held-out | AF3 trained | AF3 held-out |
|---|---|---|---|---|
| IT | 0.636 / **0.838** | 0.620 / **0.723** | 0.508 / **0.836** | 0.449 / 0.772 |
| DG-ICL-Flip | 0.803 / 0.827 | 0.692 / 0.667 | 0.546 / 0.817 | 0.534 / **0.780** |
| DG-ICL-Sym | **0.820** / 0.756 | **0.703** / 0.633 | **0.638** / 0.762 | **0.663** / 0.766 |

## 4. The mechanism, and why it is testable

**It is not label-form decay.** Decomposing canonical errors from the E0b dumps:

| condition | dataset | errors | invalid form | wrong-but-valid |
|---|---|---|---|---|
| DG-ICL-Sym | CREMA-D | 70 | 0 (0.0%) | 70 (100%) |
| DG-ICL-Sym | MELD | 383 | 8 (2.1%) | 375 (97.9%) |
| IT | CREMA-D | 61 | 0 (0.0%) | 61 (100%) |
| IT | MELD | 306 | 37 (12.1%) | 269 (87.9%) |

Sym produces well-formed, in-inventory labels and picks the wrong category. It
is *better* than IT at label form. So the loss is in category discrimination.

**The control that identifies the cause.** On the three trained tasks, Flip
holds canonical almost exactly (-0.019, -0.013, -0.001) while Sym drops 0.102,
0.035, 0.109. The structural difference between them is the target vocabulary:
Flip's targets are real English words, permuted; Sym's are synthetic two-token
strings. So the hypothesis is that **never producing real label words as targets
degrades the canonical decision boundary.**

**Existence proofs that the cost is removable.** AF3 held-out Sym scores 0.766
canonical against IT's 0.772, a loss of 0.006 while gaining 0.214 symbolic. One
cell where the trade-off simply does not appear. And Flip shows real-word
targets preserve canonical without abandoning surrogates.

## 5. Hypotheses

- **H1.** Exposing the model to real label words as targets during surrogate
  training preserves canonical discrimination without costing symbolic
  generalisation.
- **H2.** Plain data mixing is not sufficient: cross-entropy raises the
  probability of the assigned surface but applies no gradient against the
  pre-trained label prior, so a conditional negative is needed.
- **H3 (scoped down from the old H3).** Definitions govern what a category
  means but not which evidence decides it. Already measured, reported as a
  bounded negative result rather than a target.

## 6. Method

For each instance the legend maps every class `l` to a surface form `f(l)`,
drawn per instance from one of three regimes: canonical (`f(l) = l`), synthetic
symbol, or permuted real label.

- `y_w = f(y)`, the surface the legend assigns to the true class.
- `y_l = y`, the canonical name, **only when `f(y) != y`**.

The conditional is the point. Canonical-legend instances carry no negative, so
canonical ability is trained directly and never penalised. Surrogate-legend
instances penalise exactly one failure: emitting the trained label name instead
of the one the legend assigns. Plain SFT cannot express that, and neither can a
naive mixture, which has no negative gradient at all.

Objective: DPO with an NLL anchor, reusing `symbol_training_dpo.py`, whose
reference model is the same base with `disable_adapter_layers()`, so no second
model in memory. The optimiser is an implementation choice, not a contribution.

## 7. Experiment ladder

The first two stages exist to answer "is there an improvement at all" before any
novelty is layered on. Nothing here needs new datasets.

- **E1, naive mixture (the cheap decisive run).** SFT with the legend regime
  sampled per instance across canonical, synthetic and permuted-real. Does
  canonical recover while symbolic holds? Qwen, three seeds. If this alone
  closes the gap, the objective contribution is thin and we say so.
- **E2, conditional negatives.** The method of Section 6, on identical data to
  E1 so the comparison isolates the objective rather than the data recipe. (H2)
- **E3, mixture ablation.** Ratio of the three regimes, and whether permuted-real
  alone suffices or synthetic symbols are needed too.
- **E4, replication on AF3.** The backbone where Flip's symbolic transfer fails
  and where Sym already shows no held-out trade-off. The harder case.
- **E5, phrasing robustness (the old H4).** Train across definition phrasings,
  evaluate on **held-out** phrasings, never the four already in Chapter 1
  Table 5. Train/test overlap on the phrasing axis would invalidate the claim.

## 8. Metrics

Chapter 1's: canonical and symbolic macro-F1 on trained and held-out tasks,
`macro_f1_with_invalid`, counterfactual retention under the flipped legend.
Three seeds, Welch t-test with Hedges correction.

Plus one new metric from E0a, which Chapter 1 never reported: **inventory
adherence**, the fraction of predictions inside the prompt's label set on an
unseen task. DG-ICL-Sym scores 0.2 percent out-of-inventory against 8.6 for IT
and 10.6 for Flip. It is a direct measure of legend-following and it is free.

## 8a. The aim, and the measurement problem that constrains it

**Aim.** A definition follower that is invariant to the label surface form used
at inference: canonical names, synthetic symbols, permuted real words, or forms
never seen in training. Chapter 1 showed definitions can replace demonstrations.
Chapter 2 asks whether definition-following is a surface-form invariant
capability or an artifact of the single form a model was trained on.

**The measurement problem.** Retention under relabelling is only evidence of
instruction-following when the relabelling scheme is one the model never trained
on. DG-ICL-Flip reaches ~100% retention on AF3 held-out, but it trained on
derangements, so that evaluation is in-distribution for it. The paper correctly
labels it a matched-training ceiling, and its mediocre held-out original (0.667
on Qwen against IT's 0.723) confirms it bought no general capability. Sym's 68%
is meaningful for the opposite reason: deranged real words were never in its
training.

**This constrains our own method.** The mixture trains on synthetic symbols AND
permuted real words, so both `fresh` and `flipped` evaluations are
in-distribution for it. Any retention number it produces under those is a
matched-training ceiling and cannot support an instruction-following claim. This
invalidates the obvious evaluation, not the method.

**Consequence: hold out the relabelling TYPE, not just the task.** Evaluate every
condition under surrogate forms none of them trained on, so all are
out-of-distribution on equal footing:

| unseen type | example |
|---|---|
| digit strings | `07`, `42` |
| other-language real words | `colere`, `joie` |
| one-token or three-token synthetic | `qep`, `vfeldra` |
| mixed alphanumerics | `k7m2` |

Then "better instruction follower" becomes a conjunction, both halves necessary:

1. **Absolute original accuracy at or near IT's**, so the model is capable at all.
2. **Retention under unseen relabelling types**, so consistency is generalisation
   rather than a memorised scheme.

IT fails (2) outright at 21%. Flip fails (1) on held-out and only passes a
mismeasured (2). Sym passes (2) honestly at 68% and nearly passes (1). Whether
the mixture passes both is the open question, and the unseen-type evaluation is
the only way to answer it.

Implementation: a surrogate-type generator in `symbol_manager.py` alongside
`generate_fresh_symbols`, plus a `--surrogate_type` eval flag. Inference only on
existing checkpoints.

## 8b. Candidate novelty: the no-legend negative

Recorded 15 Nov 2026, to be pursued only if E1 shows the two surrogate regimes
compose. Positioning first, because it decides what the contribution can be.

**What is already taken.** [SymDPO][symdpo] (CVPR 2025) replaces text answers in
multimodal demonstrations with random symbols and applies DPO, so that LMMs stop
following text patterns and consult the image. "Symbols plus DPO for multimodal
ICL" is therefore not available as a contribution. But its mechanism *requires*
demonstrations: strip them and there is nothing to symbolise. DG-ICL is zero-shot
with definitions and no exemplars, so the preference signal has to be built from
something else entirely. That is the opening.

**The proposal.** Use the model's own no-legend prediction as the negative.

`--no_legend` already strips definitions and leaves only the label surface, and
Chapter 1 Figure 2 shows the model reverts to prior bias when it is used
(0.248 to 0.336). So the no-legend prediction is, empirically, the prior-driven
answer for that instance. For each training instance:

- `y_w = f(y)`, the surface the definition assigns.
- `y_l` = the model's answer on the same audio with the legend stripped, kept
  only when it differs from `y_w`.

This trains "answer from the definition, not from what you would have guessed
without it".

**Why it fits this setting specifically**

- *Definition-native*: the negative is defined by the presence or absence of
  definitions, so it has no demonstration analogue and cannot be expressed in
  SymDPO's formulation.
- *Zero-shot and automatic*: no exemplars, no annotation. One extra forward pass
  per instance, mineable offline once per epoch.
- *Targeted at the diagnosed failure*: Section 4 concluded the canonical loss is
  prior interference re-engaging when labels are familiar. This penalises exactly
  that, and only when the prior actually disagrees.

**Relation to contrastive decoding.** [Context-aware decoding][cad] and the CDD
variant in the thesis plan subtract no-context logits at *decode* time. Using the
same contrast as a *training* signal appears unclaimed for definition-based
prompting. Verify with a literature check before writing it up.

**Risks**

| Risk | Note |
|---|---|
| Negative mining cost | One forward pass per instance per epoch; cheap for two-token outputs, but it doubles the data pass |
| Prior-driven answer is sometimes correct | Filtering on `y_l != y_w` handles the common case; if the no-legend answer is right for the wrong reason we would penalise a correct label |
| Overlaps the conditional negative in Section 6 | They are alternative negatives for the same objective; E2 should compare them rather than assume one |

## 8c. RL / preference direction: adversarial relabelling

Recorded 16 Nov 2026. The mixture is a data recipe and thin as a contribution.
This is the candidate that could carry the chapter, but note what is and is not
new in it.

**Already done by Chapter 1.** Assigning a class's definition a different class's
real label name is DG-ICL-Flip. Random derangement, `f in Perm(L_sem)`.

**What would be new, two parts:**

1. **Targeted rather than random relabelling.** A random derangement often
   assigns an unrelated label, where the prior barely conflicts. An adversarial
   assignment maximises conflict by giving each definition the label whose
   pre-trained meaning most opposes or is most confusable with it (positive to
   negative, anger to joy). Selected by embedding similarity, by the model's own
   confusion matrix, or by its no-legend prediction. A curriculum from random to
   adversarial is the natural extension.
2. **An explicit negative gradient.** SFT on adversarial flips still only raises
   the probability of the correct surface; nothing pushes down the prior-implied
   label. A preference objective with `y_l` = the prior-implied label expresses
   exactly that, and is what cross-entropy structurally cannot do.

**Honest limits.** Adversarial flipping sharpens one axis, prior conflict, and
does nothing for the other, unfamiliar form. Flip already shows that near-perfect
performance on the prior-conflict axis coexists with weak novel-symbol transfer.
So this is a component of a method, not the whole method, and it needs the
surface-form axis alongside it.

## 9. Novelty, stated honestly

"Train on a mixture of surrogate types" is an obvious move and thin on its own.
The contribution has to be:

1. **The diagnosis.** Why surrogate tuning costs canonical accuracy, with the
   error decomposition and the Flip control. Chapter 1 reported the trade-off;
   nobody has explained it.
2. **The conditional negative**, and evidence that it beats the naive mixture
   on identical data. If it does not, that is a real finding and the chapter
   leans on the diagnosis plus the scoped negative result instead.
3. **The bounded negative result on evidence routing**, which scopes what
   definition-following means and is novel against ALME, who measure
   instruction-level compliance in proprietary models and propose no fix.

The naive mixture must appear as a baseline, not the method, or a reviewer will
ask why preference optimisation was needed.

## 10. Risks and kill criteria

| Risk | Signal | Response |
|---|---|---|
| Naive mixture already closes the gap | E1 matches IT canonical and Sym symbolic | Report it; novelty moves to the diagnosis and the negative result |
| Conditional negative adds nothing over E1 | E2 within noise of E1 | Honest ablation; chapter rests on diagnosis plus H3 |
| Mixture costs symbolic | symbolic falls below Sym | Tune the ratio in E3; if it is a genuine frontier, report the frontier |
| Gains are Qwen-only | E4 flat on AF3 | State per backbone; AF3 already shows no held-out trade-off, so it is the weaker case by construction |
| DPO instability | chosen logprob collapses | NLL anchor is in the objective; ORPO as fallback |

## 11. Compute

One RTX A6000 per run. Training is the same shape as Chapter 1: three tasks,
LoRA, about 2 to 3 days per stage for Qwen at three seeds, AF3 similar.
Evaluation reuses the existing inference path.

## 12. What was dropped, and why

- **Evidence routing as a chapter target.** E0b showed a hard null with a
  decisive control. Retained only as a scoped negative result.
- **RL or learned search over symbol choice.** Verbalizer-search literature
  reports modest gains at cost scaling with class count; RLPrompt shows the
  optimum is gibberish; our own easy/hard ablation found no effect and D-SPO
  never became a chapter.
- **CD-DPO vs CD-ORPO as a headline.** Implementation note.
- **"Beat everything" target tables.** Replaced by Pareto claims against a
  named baseline per cell.

## 13. Open questions

1. Does the canonical loss track class count (HVB 18-way and MInDS-14 14-way
   are the worst trained-task cells) or task type? That changes what the
   mixture needs to cover.
2. Is RAVDESS-Song's -0.140 the same phenomenon or a paralinguistic effect?
   It is the largest single gap and it is a held-out emotion task.
3. Target venue, given the chapter is now diagnosis plus method plus a negative
   result rather than a single new mechanism.
## 14. References

[ctxdpo]: https://openreview.net/pdf/2da95227c6c14c9a45c6a4a068648aca1bc93c37.pdf
[knowpo]: https://ojs.aaai.org/index.php/AAAI/article/view/34783/36938
[rpo]: https://arxiv.org/pdf/2501.13726
[alme]: https://arxiv.org/abs/2602.11488
[circuits]: https://www.alphaxiv.org/abs/2610.25599
[cad]: https://arxiv.org/pdf/2606.16472
[symdpo]: https://arxiv.org/abs/2411.11909

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

**Symbols with preference optimisation (closest prior work)**
- SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization (CVPR 2025). https://arxiv.org/abs/2411.11909
  Replaces demonstration answers with symbols and applies DPO so LMMs consult
  visual context. Requires demonstrations, which DG-ICL does not use, and does
  not address canonical-accuracy preservation.

**Label-space forgetting and label-name sensitivity (closest to our problem)**
- Preserving Zero-shot Capability in Supervised Fine-tuning for Multi-label Text Classification (Chen et al., Findings NAACL 2025). https://aclanthology.org/2025.findings-naacl.315/

  Reports the same problem in text: supervised fine-tuning erodes zero-shot
  ability on unseen labels EVEN WITH label descriptions, with label-embedding
  models degrading to zero as fine-tuning proceeds.

  The overlap is at the PROBLEM level, not the contribution level, so this is
  supporting evidence rather than a novelty threat. We are not claiming to have
  discovered that fine-tuning erodes unseen-label ability; we are proposing a
  training approach that avoids it. Cite it to establish the problem is real and
  unsolved for generative models.

  Their remedy is architectural and does not transfer: a dual-encoder retrieval
  setup with the label side frozen ("one-sided fine-tuning" plus a
  self-supervised auxiliary loss). A generative decoder-only speech-text LLM has
  no separate label tower to freeze. Ours is also speech rather than text, and
  generative rather than retrieval.

- When Does Supervised Fine-Tuning Reduce Instruction Sensitivity? https://arxiv.org/pdf/2608.26661
  Adjacent to the instruction-following framing in Section 8a.

- Systematic Design Choices for Fine-Tuning Text Classifiers (Lancaster, ASOC-D-25-07347). https://eprints.lancs.ac.uk/id/eprint/234483/1/ASOC-D-25-07347.pdf

  Asks whether fine-tuned decoder models use label MEANING or exploit positional
  shortcuts.

  Largely inapplicable to our setup. The legend lists categories in a fixed
  order, but which category is correct varies with the audio, so no position is
  reliably the answer and a positional strategy cannot score above chance. The
  concern applies where the answer's position is predictable, which ours is not.
  Noted for completeness in case a reviewer raises it; the unseen-surrogate-type
  evaluation would also rule it out.

- A Benchmark on Extremely Weakly Supervised Text Classification. https://arxiv.org/pdf/2305.12749
  Prompt-based methods are highly unstable when label words are swapped for
  similar-meaning alternatives. Establishes label-name sensitivity as a known
  problem, in text.

- Text Classification Using Label Names Only (LOTClass, EMNLP 2020). https://aclanthology.org/2020.emnlp-main.724.pdf
- Label-semantics Aware Generative Approach for Domain-Agnostic Multilabel Classification. https://arxiv.org/html/2506.06806v1

  Searched for and did NOT find (as of 16 Nov 2026): any work testing whether a
  model trained with definitions generalises to label surface forms it never saw,
  in any modality. No surface-form invariance evaluation, nothing in speech. That
  specific claim appears unoccupied and is what Section 8a targets.

**Forgetting, anchoring and the alignment tax (relevant to the objective)**
- Anchored Supervised Fine-Tuning. https://arxiv.org/html/2509.23753v3
- Proximal Supervised Fine-Tuning. https://arxiv.org/pdf/2508.17784
- Stabilizing LLM Supervised Fine-Tuning via Explicit Distributional Control. https://arxiv.org/html/2605.04468v1
- Mitigating the Alignment Tax of RLHF. https://arxiv.org/pdf/2309.06256
- Preference Data Selection for Mitigating the Alignment Tax in LLMs. https://arxiv.org/pdf/2608.24192
- UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function. https://arxiv.org/html/2410.21438v1

  Note: these establish that a KL/reference anchor mitigates forgetting in text
  LLMs. "We used a reference-anchored objective and it forgot less" is therefore
  NOT a contribution on its own. What is ours is the phenomenon (surrogate
  tuning degrading canonical category discrimination in speech-text LLMs), the
  claim that the trade-off is an artifact of the objective, and any mechanism
  tailored to it.

**Our own prior work**
- Chapter 1: Definition-Guided In-Context Learning for Zero-Shot Spoken Language Understanding (under review).
- Spoken Language Understanding on Unseen Tasks With In-Context Learning. https://arxiv.org/html/2505.07731
