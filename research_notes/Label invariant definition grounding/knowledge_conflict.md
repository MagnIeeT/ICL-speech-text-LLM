# Knowledge Conflicts, Context Faithfulness vs Parametric Priors: Training Methods That Get Both

Framing for the thesis: parametric prior = semantics of the label name (e.g. "positive", "angry"); context = the label definitions in the prompt. The "original-label" model behaves like a stubborn, prior-following LLM under context-memory conflict (25% retention under swapped names); the "permuted-label" model behaves like a context-faithful model that has partly lost the benefit of the prior (95% retention, lower original-label accuracy). Every item below is mapped to that trade-off.

Verification note: items whose arXiv id / venue were confirmed in this session via arXiv, ACL Anthology, CVF or NeurIPS pages carry a link. Items marked **[unverified this session]** rely on prior knowledge and should be checked before citing.

## Q1. Surveys and foundational framing (Xu et al. 2024; Longpre et al. 2021)

### Takeaway
The knowledge-conflict literature formalises exactly the thesis's failure mode as "context-memory conflict", and its founding paper (Longpre et al.) already shows the standard fix (train on entity-substituted counterfactual contexts) is the same idea as permuted-label training. A key 2025 result (context-parametric inversion) explains WHY ordinary instruction tuning on original labels erodes context reliance: examples where context agrees with the prior teach the model it can ignore context.

### Cited Findings
- **Xu, Qi, Guo, Wang, Wang, Zhang, Xu. "Knowledge Conflicts for LLMs: A Survey." EMNLP 2024 (main), pp. 8541-8565, arXiv:2403.08319.** Taxonomy: context-memory conflict, inter-context conflict, intra-memory conflict; reviews causes, model behaviour and solutions. Repo: pillowsofwind/Knowledge-Conflicts-Survey. ([ACL Anthology](https://aclanthology.org/2024.emnlp-main.486/); [arXiv](https://arxiv.org/abs/2403.08319))
  - Relevance: the thesis setting is a context-memory conflict where "memory" is the lexical semantics of the label token, not a world fact.
- **Longpre, Perisetla, Chen, Ramesh, DuBois, Singh. "Entity-Based Knowledge Conflicts in Question Answering." EMNLP 2021, pp. 7052-7063, arXiv:2109.05052.** Builds conflicts by substituting the answer entity throughout the context (person, date, numeric, org, location types via spaCy NER + Wikidata). Finds QA models over-rely on memorised parametric knowledge even when context is correct. Code: apple/ml-knowledge-conflicts. ([ACL Anthology](https://aclanthology.org/2021.emnlp-main.565); [NSF PAR](https://par.nsf.gov/servlets/purl/10462823); [ar5iv](https://ar5iv.labs.arxiv.org/html/2109.05052))
  - The paper also proposes training with substituted examples as a mitigation that reduces memorisation and improves OOD generalisation **[exact numbers unverified this session]**.
- **Goyal, Baek, Kolter, Raghunathan. "Context-Parametric Inversion: Why Instruction Finetuning May Not Actually Improve Context Reliance." ICLR 2025 (Oral), arXiv:2410.10796.** During instruction finetuning, context reliance under knowledge conflict first rises and then falls, while standard benchmark performance keeps improving. Observed on TULU, Alpaca, UltraChat with Llama, Mistral, Pythia. Cause: finetuning examples whose context *agrees* with parametric knowledge (the model can answer them without reading context). Mitigations proposed give "limited but insightful gains". ([arXiv](https://arxiv.org/abs/2410.10796); [OpenReview PDF](https://openreview.net/pdf?id=SPS6HzVzyt))
- **Wei et al. "Larger language models do in-context learning differently." arXiv:2303.03846 (2023).** Flipped-label and semantically-unrelated-label ICL; instruction tuning strengthens reliance on semantic priors of label names. **[unverified this session; cited indirectly by the symbol-tuning paper below]** ([symbol tuning paper discussing it](https://arxiv.org/abs/2305.08298))
- **Xie, Zhang, Chen, Lou, Su. "Adaptive Chameleon or Stubborn Sloth." ICLR 2024, arXiv:2305.13300.** LLMs are receptive to coherent, convincing conflicting evidence but show confirmation bias when evidence is mixed. **[unverified this session]**

### Inferences
- The 25% retention of the original-label model is the expected outcome under context-parametric inversion: when trained on original labels, every training example has "context agrees with prior" (the definition matches the name), so the gradient never rewards reading the definition. This is a direct, citable mechanism for the thesis's observation.
- The permuted-label model is the analogue of Longpre-style substitution training / KAFT counterfactual augmentation.

### Gaps
- No survey found that treats label-name semantics in classification as a "parametric prior" in the knowledge-conflict sense; this framing appears novel and worth stating explicitly in the thesis.

## Q2. Training methods for context faithfulness, and whether they keep prior-use when context agrees

### Takeaway
The best-supported recipe for "both" is a **mixture** of factual (agreeing), counterfactual (conflicting) and irrelevant contexts (KAFT), which reported large controllability gains while matching standard performance on regular contexts. Preference-based (Context-DPO, KnowPO) and RL (CANOE) methods push faithfulness further but mostly report generic capability retention, not accuracy on agreeing contexts specifically. DisentQA and FaithfulRAG are the methods that explicitly keep the prior alive as a separate signal rather than suppressing it.

### Cited Findings

**KAFT: knowledge-aware finetuning**
- **Li, Rawat, Zaheer, Wang, Lukasik, Veit, Yu, Kumar. "Large Language Models with Controllable Working Memory." Findings of ACL 2023, pp. 1774-1793, arXiv:2211.05110.** Defines *controllability* (prioritise relevant context that conflicts with memory) and *robustness* (ignore irrelevant context, fall back to memory). KAFT augments standard supervised data with counterfactual contexts and irrelevant contexts (alongside factual ones). Pretrained and finetuned T5/PaLM show low controllability that does not improve with scale. KAFT PaLM-540B: about 24x better controllability than noisy finetuning under conflict, while performing similarly on regular contexts; most of the gain comes from counterfactual augmentation. ([ACL Anthology](https://aclanthology.org/2023.findings-acl.112); [arXiv](https://arxiv.org/abs/2211.05110v1))
  - Achieves both? **Yes, as reported**: conflict-following up, regular-context performance roughly unchanged. Irrelevant-context arm also teaches when to use the prior.
  - Relevance: **highest**. Direct template: train on a mix of (a) original labels + correct definitions (agreeing), (b) permuted names + definitions (conflicting), (c) possibly definitions that are uninformative, so the model must use the name prior. The thesis's two models are KAFT's two arms trained separately.

**DisentQA**
- **Neeman, Aharoni, Honovich, Choshen, Szpektor, Abend. "DisentQA: Disentangling Parametric and Contextual Knowledge with Counterfactual Question Answering." ACL 2023, pp. 10056-10070, arXiv:2211.05655.** Single model outputs two answers: a contextual answer and a parametric answer. Counterfactual examples built by swapping answer mentions in context (following Longpre et al.); plus answerability augmentation (abstain when context lacks the answer). More robust to conflicts while still producing useful disentangled answers; parametric answers still leak context (imperfect disentanglement). ([ACL Anthology](https://aclanthology.org/2023.acl-long.559); [arXiv](https://arxiv.org/pdf/2211.05655))
  - Achieves both? **By design**: keeps the prior as an explicit output rather than suppressing it.
  - Relevance: high. A speech analogue would be a model that predicts both "label by definition" and "label by name", making the conflict explicit and supervisable; at test time output the definition-grounded one.

**Context-faithful prompting (training-free)**
- **Zhou, Zhang, Poon, Chen. "Context-faithful Prompting for Large Language Models." Findings of EMNLP 2023, arXiv:2303.11315.** Opinion-based prompts (frame context as a narrator's statement, ask for the narrator's opinion) and counterfactual demonstrations; significant faithfulness gains on MRC and relation extraction under knowledge conflict and abstention; no training. ([arXiv](https://arxiv.org/abs/2303.11315))
  - Relevance: cheap baseline. "According to the definitions given, which class..." style prompting, and in-context counterfactual (permuted-label) demonstrations.

**Context-DPO**
- **Bi et al. "Context-DPO: Aligning Language Models for Context-Faithfulness." Findings of ACL 2025, pp. 10280-10300, arXiv:2412.15280.** DPO with chosen = faithful answer grounded in counterfactual context, rejected = "stubborn" answer from real-world fact. Introduces ConFiQA benchmark. Reports 35% to 280% faithfulness gains on open-source models; claims generative capability preserved (TruthfulQA). Code: byronBBL/Context-DPO. ([arXiv](https://arxiv.org/abs/2412.15280); [ACL Anthology](https://aclanthology.org/2025.findings-acl.536))
  - Achieves both? **Partly**: preserved general capability is reported, but accuracy on agreeing contexts is not the headline metric.
  - Relevance: high and directly portable. Preference pair per permuted example: chosen = definition-consistent label, rejected = name-prior label. This penalises exactly the 75% failure mode of the original-label model while leaving the agreeing case unaffected (chosen and prior coincide there, so no pair exists).

**KnowPO**
- **Zhang et al. "KnowPO: Knowledge-aware Preference Optimization for Controllable Knowledge Selection in Retrieval-Augmented Language Models." AAAI 2025 (DOI 10.1609/aaai.v39i24.34783), arXiv:2408.03297.** Names two failure modes from plain finetuning: *contextual ignorance* (ignoring context) and *contextual overinclusion* (over-relying on irrelevant/noisy context). Builds preference data simulating retrieval with noise categorised by relevance; DPO after instruction tuning; rewriting strategy and data-ratio adjustment to fix length and behaviour-pattern imbalance in pairs. Reports >37% over prior conflict-handling methods and OOD generalisation. ([arXiv](https://arxiv.org/abs/2408.03297); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34783))
  - Achieves both? **Targets both explicitly** (the two failure modes are the two sides of the thesis trade-off).
  - Relevance: high. The "data-ratio adjustment" lesson maps to choosing the mix of original vs permuted label episodes.

**CANOE (RL)**
- **Si et al. "Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning." AAAI 2026, arXiv:2505.16483.** Synthesises short-form QA with verifiable answers; Dual-GRPO with three rule-based rewards; reasoning then long-form then short-form answer. Evaluated on 11 tasks; +22.6% overall for Llama3-Instruct-8B; claims to exceed GPT-4o on overall faithfulness score. ([arXiv](https://arxiv.org/abs/2505.16483); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40582))
  - Relevance: medium. Verifiable rewards are trivially available in classification (label correct under the given definitions), so GRPO over mixed original/permuted episodes is feasible.

**FaithfulRAG**
- **Zhang et al. "FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation." ACL 2025 (long), pp. 21863-21882, arXiv:2506.08938.** Argues prior faithful-RAG methods enforce adherence by sidelining parametric knowledge, which increases misreading of context: in their preliminary study, those methods cut unfaithful errors by about 6.65% but added about 6.42% incorrect-match errors. Method: self-fact mining (elicit the model's own fine-grained knowledge), identify fact-level conflicts with context, then self-thinking reasoning before answering. Notes that fine-tuning approaches (KAFT, SFR-RAG) are compute-heavy and transfer poorly. ([arXiv](https://arxiv.org/abs/2506.08938v2); [ACL Anthology](https://aclanthology.org/2025.acl-long.1062))
  - Achieves both? **Its core claim** is that suppressing the prior costs accuracy, and that explicitly surfacing and reconciling it is better.
  - Relevance: high conceptually. This is the published version of the thesis's observation that the permuted-label model loses accuracy on original labels. An analogue: have the model first state what the label name usually means, compare to the given definition, then decide.

**Other 2024-2026 methods surfaced (less verified)**
- Conflict-Aware Soft Prompting for RAG (EMNLP 2025 main): trains a separate context assessor and keeps the LLM frozen, motivated by the risk that finetuning erodes parametric knowledge and generalisation. ([ACL Anthology PDF](https://aclanthology.org/2025.emnlp-main.1371.pdf)) Authors and arXiv id not captured.
- CLEAR (2025 preprint, probing latent knowledge conflict): conflict-aware finetuning with an auxiliary attention loss. ([arXiv](https://arxiv.org/html/2510.12460v1)) Details not verified beyond search snippet.
- Faithfulness-QA (2026, arXiv:2604.25313): ~99k counterfactual entity-substitution samples for training context-faithful RAG; argues only conflicting contexts can separate genuine context-following from incidental agreement. ([arXiv](https://arxiv.org/html/2604.25313v2))
- Prior knowledge only **[unverified this session]**: Parenting (Zhong et al. 2024, arXiv:2410.10360, parameter decoupling for knowledge selection); JUICE / superposition heads (Li et al., ICML 2025, arXiv:2503.10996, head-level intervention); PH3 "Cutting off the head ends the conflict" (Jin et al., ACL 2024 Findings, arXiv:2402.18154, prune memory vs context heads).

### Inferences
- The literature converges on: training only on counterfactual/permuted data yields faithfulness but costs accuracy where the prior is useful (FaithfulRAG's 6.42% incorrect-match increase is the RAG version of the thesis's original-label drop); training only on agreeing data yields stubbornness (context-parametric inversion). KAFT's mixture plus an explicit preference signal (Context-DPO / KnowPO) is the most direct "both" recipe.
- A clean thesis experiment: mixed training (original + permuted episodes at a tuned ratio) and optionally a DPO stage where rejected = name-prior label on permuted episodes. Report both original-label accuracy and retention, as KAFT reports both regular and counterfactual performance.

### Gaps
- Few papers report accuracy on *agreeing* contexts as a first-class metric after faithfulness training; Goyal et al. and the Faithfulness-QA paper note this gap. KAFT's "similar on regular contexts" is the clearest such number found.
- Exact KAFT mixing ratios and per-dataset numbers were not extracted.

## Q3. Decoding-time methods (CAD and adaptive variants)

### Takeaway
CAD amplifies the difference between with-context and without-context distributions and strongly helps under conflict, but a fixed contrast hurts when there is no conflict; AdaCAD fixes this by scaling the contrast with the measured conflict (JSD), which is the decoding-time version of "use the prior when it agrees, follow context when it doesn't".

### Cited Findings
- **Shi, Han, Lewis, Tsvetkov, Zettlemoyer, Yih. "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding." NAACL 2024 (short), pp. 783-791, arXiv:2305.14739.** Contrastive output distribution, in the paper's form roughly p ∝ p(y|c,x)^(1+α) / p(y|x)^α. Training-free; works across OPT, GPT, LLaMA, FLAN-T5; +14.3% factuality for LLaMA on summarisation; "especially effective" at overriding conflicting prior knowledge. ([ACL Anthology](https://aclanthology.org/2024.naacl-short.69); [arXiv](https://arxiv.org/abs/2305.14739)) Formula written from prior knowledge, not re-read this session.
- **Wang, Prasad, Stengel-Eskin, Bansal. "AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge." NAACL 2025 (long), pp. 11636-11652, arXiv:2409.07394.** Replaces CAD's fixed α with per-token weight given by Jensen-Shannon divergence between with- and without-context distributions; contrast shrinks to normal decoding when conflict is small. States CAD works when conflict is uniformly high but struggles when conflict varies across examples. Outperforms decoding baselines across 4 LLMs, 6 QA and 3 summarisation datasets. ([arXiv](https://arxiv.org/abs/2409.07394v2); [ACL Anthology](https://aclanthology.org/2025.naacl-long.581))
- Follow-ups surfaced: CoCoA, confidence- and context-aware adaptive decoding (arXiv:2508.17670) ([arXiv](https://arxiv.org/html/2508.17670v1)); "From Context-Aware to Conflict-Aware" generalising contrastive decoding (arXiv:2606.10298) ([arXiv](https://arxiv.org/pdf/2606.10298)). Not read in detail.
- **COIECD** (Yuan et al., ACL 2024, arXiv:2402.11893): adaptive decoding with a contextual-information-entropy constraint that applies contrast only when conflict is detected, aimed at preserving performance when no conflict exists. **[unverified this session]**

### Inferences
- Direct thesis analogue: CAD over label scoring with "with definitions" vs "without definitions" (name only) prompts. For permuted labels, the contrast removes the name prior; AdaCAD's JSD gating should leave original-label accuracy intact because with/without-definition distributions agree there. This is cheap to test on the existing original-label model and could be a strong baseline or even a component.
- A training-time analogue of AdaCAD is to weight a faithfulness loss by the divergence between definition-conditioned and name-only predictions.

### Gaps
- No CAD-style numbers found for classification with label-definition prompts or for speech LLMs.

## Q4. ICL-specific analogue: label-name priors (closest to the thesis)

### Takeaway
Symbol tuning is the direct ICL precedent for the thesis's permuted-label training: replacing natural-language labels with arbitrary symbols forces reliance on in-context mappings, and restores flipped-label following that instruction tuning removes.

### Cited Findings
- **Wei, Hou, Lampinen, Chen, Huang, Tay, Chen, Lu, Zhou, Ma, Le. "Symbol tuning improves in-context learning in language models." EMNLP 2023 (main), arXiv:2305.08298.** Finetune on ICL examples whose natural-language labels are replaced by arbitrary symbols. Symbol-tuned Flan-PaLM (8B to 540B) follows flipped labels much better than instruction-tuned models at all sizes; instruction tuning had degraded the flipped-label ability that pretrained-only models partly had, and symbol tuning restores it roughly to pretrained levels. Flipped-label ability was not trained directly. Caveat: flipped-label performance on average not significantly better than random. ([ACL Anthology](https://aclanthology.org/2023.emnlp-main.61); [arXiv](https://arxiv.org/abs/2305.08298v1); [Google Research blog](https://research.google/blog/symbol-tuning-improves-in-context-learning-in-language-models/))
  - Authors list is from prior knowledge **[partly unverified]**; the venue and id are verified.
  - Relevance: **highest** for framing. The thesis's permuted-label model is a definition-based variant of symbol tuning; the thesis contributes a speech setting and an explicit measurement of the original-label cost.

### Inferences
- Symbol tuning mixed with instruction-tuning data (as in the paper's data mixture) is itself a "both" recipe; worth checking whether the paper reports that symbol tuning hurts standard tasks (I did not extract this).

### Gaps
- Whether symbol tuning preserves accuracy on natural-label tasks was not verified in this session.

## Q5. Multimodal analogues: language prior in VQA and MLLMs

### Takeaway
VQA debiasing (RUBi, LMH, CSS) mirrors the thesis trade-off almost exactly: methods that suppress the question-only prior win on VQA-CP (shifted priors) but lose in-distribution accuracy. Counterfactual VQA and especially Introspective Distillation (IntroD) were designed to recover ID accuracy while keeping OOD robustness, which is the vision analogue of "use the prior when it agrees". In MLLMs, mDPO and VCD target grounding vs language prior at preference-training and decoding time.

### Cited Findings
- **VQA-CP** (Agrawal, Batra, Parikh, Kembhavi, CVPR 2018, arXiv:1712.00377): train/test with different per-question-type answer distributions. **[id/authors unverified this session]**; benchmark design confirmed in CSS paper text ([CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Counterfactual_Samples_Synthesizing_for_Robust_Visual_Question_Answering_CVPR_2020_paper.html)).
- **RUBi** (Cadene, Dancette, Ben-younes, Cord, Parikh, NeurIPS 2019, arXiv:1906.10169): question-only branch masks the main model's logits during training so biased examples are down-weighted; removed at test time. **[unverified this session]**
- **LMH, "Don't Take the Easy Way Out"** (Clark, Yatskar, Zettlemoyer, EMNLP 2019, arXiv:1909.03683): product-of-experts with a bias-only model plus learned mixin and entropy penalty. **[unverified this session]**
- **CSS, "Counterfactual Samples Synthesizing for Robust VQA"** (Chen et al., CVPR 2020, arXiv:2003.06576): mask critical objects (V-CSS) or question words (Q-CSS) and assign different answers, creating counterfactual training samples; on top of LMH reaches 58.95% on VQA-CP v2 (+6.5). VQA v2 (ID) drop not extracted. ([CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Counterfactual_Samples_Synthesizing_for_Robust_Visual_Question_Answering_CVPR_2020_paper.html); [arXiv](https://arxiv.org/pdf/2003.06576))
  - Relevance: CSS is the closest VQA analogue of permuted-label training (synthesised counterfactuals where the prior gives the wrong answer).
- **Counterfactual VQA (CF-VQA)** (Niu, Tang, Zhang, Lu, Hua, Wen, CVPR 2021, pp. 12700-12710, arXiv:2006.04315): treats language bias as the direct effect of question on answer; estimates it counterfactually by blocking the multimodal path and subtracts natural direct effect from total effect at inference. Keeps the language prior in training and removes only its direct effect at test time. ([CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Niu_Counterfactual_VQA_A_Cause-Effect_Look_at_Language_Bias_CVPR_2021_paper.html); [arXiv](https://arxiv.org/abs/2006.04315v4)) Full author list from prior knowledge.
- **IntroD, "Introspective Distillation for Robust Question Answering"** (Niu, Zhang, NeurIPS 2021, arXiv:2111.01026): states that debiasing methods gain OOD accuracy with a considerable sacrifice of ID accuracy, so are only usable when the test distribution is known. IntroD blends an ID teacher and an OOD teacher per sample by introspecting whether the sample fits the factual ID world or the counterfactual OOD world. Reported: CSS+IntroD reaches 60.17% on VQA-CP v2 test (vs CSS 58.95%) while recovering ID accuracy (LMH 58.38 to 63.68 on the ID evaluation; table labelling ambiguous). Also validated on SQuAD position bias. Ablations attribute gains to causal introspection, not simple ensembling. ([NeurIPS proceedings](https://proceedings.neurips.cc/paper/2021/hash/878d5691c824ee2aaf770f7d36c151d6-Abstract.html); [arXiv](https://arxiv.org/pdf/2111.01026))
  - Achieves both? **Yes, explicitly the goal.**
  - Relevance: **very high**. Direct recipe: distil a single student from the original-label model (ID teacher, strong prior) and the permuted-label model (OOD teacher, definition-grounded), weighting per example by which teacher fits. The thesis already has both teachers.
- **Teney et al. "On the Value of Out-of-Distribution Testing: An Example of Goodhart's Law." NeurIPS 2020, arXiv:2005.09241.** Most VQA-CP methods exploit knowledge of how the OOD split was built, use the OOD test set for model selection, and report ID performance after retraining on VQA v2; trivial methods (including random answering on some question types) beat SOTA. Recommends reporting ID performance on held-out VQA-CP training data. ([arXiv](https://arxiv.org/abs/2005.09241); [NeurIPS](https://neurips.cc/virtual/2020/poster/16833))
  - Relevance: methodological warning. Select checkpoints on a held-out split, not on the permuted-label test set, and report original and permuted performance from the same model.
- **mDPO** (Wang, Zhou, Huang, Xu, Zhang, Poon, Chen. "mDPO: Conditional Preference Optimization for Multimodal Large Language Models." EMNLP 2024 main, arXiv:2406.11839): identifies the *unconditional preference problem*, where multimodal DPO learns language-only preferences and ignores the image. Adds an image-conditional preference term (prefer the response given the true image over a corrupted image) and a reward anchor keeping chosen rewards positive. Gains across two MLLMs and three benchmarks with reduced hallucination. ([arXiv](https://arxiv.org/abs/2406.11839))
  - Relevance: high. Conditional preference maps to "prefer the label given the true definitions over the label given corrupted/swapped definitions", forcing the model to condition on definitions rather than names.
- **VCD** (Leng et al., "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding," CVPR 2024, pp. 13872-13882, arXiv:2311.16922): contrasts outputs from the original and a distorted (noised/masked) image to counter language priors; up to +7.4 F1 on POPE; training-free. Later work (ClearSight) notes added inference cost and that subtracting language prior can hurt coherence. ([CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html); [ClearSight](https://arxiv.org/pdf/2503.13107))
  - Relevance: the audio analogue (contrast with noised/absent audio) targets a different prior (text over audio); for the label problem the right "distortion" is removing or scrambling definitions, i.e. CAD.

### Inferences
- The VQA debiasing history is an existence proof that "suppress the prior" methods trade ID for OOD, and that the field moved to methods that keep the prior and gate it (CF-VQA, IntroD). The thesis can cite this trajectory to motivate a gated or distilled method instead of pure permuted training.
- Ranked candidate methods for label-invariant definition grounding, by fit: (1) KAFT-style mixture of original and permuted episodes; (2) IntroD-style distillation from the existing two models; (3) Context-DPO / mDPO-style conditional preference on permuted vs true definitions; (4) AdaCAD-style JSD-gated contrastive decoding as a training-free baseline; (5) DisentQA-style dual outputs.

### Gaps
- ID (VQA v2) accuracy costs for RUBi, LMH and CSS were not extracted this session; IntroD's tables are the best single source for these numbers.
- No speech or audio-LLM paper was found that applies knowledge-conflict or symbol-tuning methods to label definitions; this appears to be an open niche.
