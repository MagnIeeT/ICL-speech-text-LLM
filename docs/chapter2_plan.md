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
