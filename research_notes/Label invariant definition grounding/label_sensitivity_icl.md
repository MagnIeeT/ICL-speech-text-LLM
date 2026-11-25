# Label surface-form sensitivity and label-name effects in ICL and instruction tuning

Verification note: items marked VERIFIED-THIS-SESSION had their arXiv or venue page fetched or surfaced by search during this research pass. Items marked FROM-PRIOR-KNOWLEDGE are well-known papers whose arXiv id and venue I am confident of but did not re-open in this pass; spot-check them. Items marked UNVERIFIED have an uncertain detail (usually venue or author list), stated explicitly.

## Q1. How do label names, flipped labels, and symbol labels affect ICL predictions?

### Takeaway
ICL predictions come from (a) semantic priors attached to label tokens and task format ("task recognition") and (b) the input-label mapping in the prompt ("task learning"). Small and instruction-tuned models lean on (a), so label names dominate; only large models, or models trained with symbol tuning, override priors with flipped or semantically unrelated labels. A definition-grounded classifier therefore has to be explicitly trained or constrained to ignore label surface form.

### Cited Findings
- Min, Lyu, Holtzman, Artetxe, Lewis, Hajishirzi, Zettlemoyer, "Rethinking the Role of Demonstrations", EMNLP 2022, arXiv 2202.12837. Random labels in demos barely hurt across 12 models; label space, input distribution and format matter most. Relevance: the label SPACE carries much of the signal. FROM-PRIOR-KNOWLEDGE ([arXiv](https://arxiv.org/abs/2202.12837)).
- Yoo et al., "Ground-Truth Labels Matter: A Deeper Look into Input-Label Demonstrations", EMNLP 2022, arXiv 2205.12685. With Label-Correctness Sensitivity and Ground-truth Label Effect Ratio metrics, correct labels matter substantially and task-dependently; sensitivity correlates with task complexity. FROM-PRIOR-KNOWLEDGE; correlation claim restated in [ACL 2025 paper](https://aclanthology.org/2025.acl-long.196/).
- Wei, Wei, Tay, Tran, Webson, Lu, Chen, Liu, Huang, Zhou, Ma, "Larger language models do in-context learning differently", arXiv 2303.03846 (2023). Flipped-label ICL and semantically-unrelated-label ICL (SUL-ICL). Large models override semantic priors when shown flipped labels (emergent); small models cannot and drop more under SUL-ICL; instruction tuning strengthens reliance on semantic priors. Venue: arXiv only (UNVERIFIED venue) ([arXiv](https://arxiv.org/abs/2303.03846)).
- Wei, Hou, Lampinen, Chen, Huang, Tay, Chen, Lu, Zhou, Ma, Le, "Symbol tuning improves in-context learning in language models", EMNLP 2023, arXiv 2305.08298. Fine-tune Flan-PaLM (up to 540B) on exemplars whose labels are replaced by arbitrary symbols, instructions removed. Gains on unseen ICL tasks when instructions or meaningful label names are absent, large gains on algorithmic reasoning, restores flipped-label following lost via instruction tuning. Appendix: per-exemplar random symbols (no consistent mapping) generally worse than standard symbol tuning. VERIFIED-THIS-SESSION ([arXiv](https://arxiv.org/html/2305.08298v2)).
- Wang, Li, Dai, Chen, Zhou, Meng, Zhou, Sun, "Label Words are Anchors", EMNLP 2023 (best long paper), arXiv 2305.14160. Label-word tokens aggregate demo information in shallow layers; deep layers read from them. Proposes anchor re-weighting, demo compression, error diagnosis. Relevance: label tokens are a mechanistic bottleneck; a diagnostic for chapter 2 is whether information flows through definition tokens instead. FROM-PRIOR-KNOWLEDGE ([arXiv](https://arxiv.org/abs/2305.14160)).
- Pan, Gao, Chen, Chen, "What In-Context Learning 'Learns' In-Context: Disentangling Task Recognition and Task Learning", Findings ACL 2023, arXiv 2305.09731. TR works with random labels and scales little; TL (abstract/symbol labels) emerges only at scale. Symbol labels underperform, attributed to unnaturalness in pretraining. FROM-PRIOR-KNOWLEDGE ([arXiv](https://arxiv.org/abs/2305.09731)).
- Kossen, Gal, Rainforth, "In-Context Learning Learns Label Relationships but Is Not Conventional Learning", ICLR 2024, arXiv 2307.12375. ICL uses in-context labels and can learn novel tasks, but cannot fully overcome pretraining label preferences and weights context unequally. FROM-PRIOR-KNOWLEDGE ([arXiv](https://arxiv.org/abs/2307.12375)).
- Webson and Pavlick, "Do Prompt-Based Models Really Understand the Meaning of Their Prompts?", NAACL 2022, arXiv 2109.01247. Irrelevant or misleading instructions learn about as fast as good ones, but target-word (label word) choice matters a lot. FROM-PRIOR-KNOWLEDGE ([arXiv](https://arxiv.org/abs/2109.01247)).
- Halawi, Denain, Steinhardt, "Overthinking the Truth", ICLR 2024, arXiv 2307.09476. Under false demos, accuracy diverges after a "critical layer"; "false induction heads" copy wrong labels. FROM-PRIOR-KNOWLEDGE ([arXiv](https://arxiv.org/abs/2307.09476)).
- Tang et al., "Large Language Models Can be Lazy Learners: Analyze Shortcuts in In-Context Learning", Findings ACL 2023, arXiv 2305.17256. LLMs exploit prompt shortcuts, larger models more so. FROM-PRIOR-KNOWLEDGE; authors UNVERIFIED.
- Lin and Lee, "Dual Operating Modes of In-Context Learning", ICML 2024, arXiv 2402.18819. Theory of task retrieval vs task learning. FROM-PRIOR-KNOWLEDGE.
- Chan et al., "Data Distributional Properties Drive Emergent In-Context Learning in Transformers", NeurIPS 2022, arXiv 2205.05055. Bursty data and labels whose meaning changes across contexts favor in-context over in-weights learning. Relevance: per-instance symbol training pushes models to read context. FROM-PRIOR-KNOWLEDGE.
- Li et al. (first author Shiyang Li), "Instruction-following Evaluation through Verbalizer Manipulation", Findings NAACL 2024, arXiv 2307.10558. Natural, neutral (semantically irrelevant), unnatural (flipped) verbalizers specified in the instruction. Larger models do well on natural and neutral but diverge sharply on unnatural; GPT-4 barely above random on the hardest unnatural verbalizer. Relevance: analogue of the permuted-real-label condition. VERIFIED-THIS-SESSION ([ACL Anthology](https://aclanthology.org/2024.findings-naacl.233); [arXiv](https://arxiv.org/html/2307.10558v2)).
- Krishna Kumar, "Semantic Anchors in In-Context Learning: Why Small LLMs Cannot Flip Their Labels", arXiv 2511.21038 (Nov 2025, preprint). 8 tasks, 8 models 1 to 12B: natural-demo correct predictions mostly coincide with zero-shot; inverted demos do not yield coherent anti-semantic classifiers; semantic override rate exactly zero. Overriding label semantics at this scale needs interventions beyond ICL. VERIFIED-THIS-SESSION ([arXiv](https://arxiv.org/abs/2511.21038)).
- Jang, Jang, Kweon, Jeon, Yu, "Rectifying Demonstration Shortcut in In-Context Learning", NAACL 2024, arXiv 2403.09488. "Demonstration Shortcut" = reliance on priors over demo mappings; In-Context Calibration; gains in standard and SUL "Task Learning" settings on OPT, GPT, Llama2. VERIFIED-THIS-SESSION ([arXiv](https://arxiv.org/abs/2403.09488)).
- "Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding", arXiv 2502.13738 (2025). Contrast positive vs negative demo distributions. Authors and venue UNVERIFIED ([arXiv](https://arxiv.org/abs/2502.13738)).
- "Improving Input-label Mapping with Demonstration Replay for In-context Learning", arXiv 2310.19572. Uses SUL labels to force mapping use. Authors and venue UNVERIFIED ([arXiv](https://arxiv.org/pdf/2310.19572)).
- "Interpret and Improve In-Context Learning via the Lens of Input-Label Mappings", ACL 2025. About 5% of attention heads apply input-label mappings. Authors UNVERIFIED ([ACL Anthology](https://aclanthology.org/2025.acl-long.196/)).
- Wang et al., "Can In-context Learning Really Generalize to Out-of-distribution Tasks?", arXiv 2410.09695. Abstract label symbols work if the model trained on sufficiently many similar tasks. Venue UNVERIFIED ([arXiv](https://arxiv.org/pdf/2410.09695)).
- "Spoken Language Understanding on Unseen Tasks With In-Context Learning", arXiv 2505.07731 (2025). Randomized-label fine-tuning of a speech-text LLM; reported relative few-shot gains 10.1% SLUE-VoxCeleb, 95.3% SLUE-HVB, 64.3% SLUE-VoxPopuli vs regular fine-tuning. Possibly overlaps with the thesis author's own chapter 1 work; check before citing as external ([arXiv](https://arxiv.org/abs/2505.07731)).

### Inferences
- Chapter 1 conditions map onto established probes: original labels = standard ICL; random per-instance symbols = SUL-ICL / symbol tuning; permuted real labels = flipped-label ICL / unnatural verbalizer. Literature predicts permuted real labels are hardest, especially for small or instruction-tuned backbones.
- Symbol tuning's weaker per-exemplar-random variant cautions that per-instance symbols remove both prior and stable mapping; the only signal left is the definition, which is what chapter 2 wants but needs strong supervision.

### Gaps
- No ICL paper found testing label DEFINITIONS (rather than demonstrations) under flipped or symbolic labels. Appears to be an open gap.
- Wei et al. 2303.03846 venue not confirmed.

## Q2. Calibration and label bias: inference-time fixes

### Takeaway
Calibration methods estimate a label prior (content-free input, in-domain random text, test batch, demos, option permutations) and divide it out. They reduce surface-form bias but rescale priors over named labels rather than grounding decisions in definitions.

### Cited Findings
- Zhao, Wallace, Feng, Klein, Singh, "Calibrate Before Use", ICML 2021, arXiv 2102.09690. Majority-label, recency, common-token bias; contextual calibration with content-free "N/A" input. FROM-PRIOR-KNOWLEDGE ([arXiv](https://arxiv.org/abs/2102.09690)).
- Holtzman, West, Shwartz, Choi, Zettlemoyer, "Surface Form Competition", EMNLP 2021, arXiv 2104.08315. Surface forms of the same answer compete; Domain Conditional PMI fixes. FROM-PRIOR-KNOWLEDGE.
- Fei, Hou, Chen, Bosselut, "Mitigating Label Biases for In-context Learning", ACL 2023, arXiv 2305.19148. Vanilla-label, context-label, domain-label bias; domain-context calibration with random in-domain words. GPT-J prior shifts to 0.95 "hate" vs 0.05 "neutral" given in-domain random words. Up to 37% Macro-F1 gains. VERIFIED-THIS-SESSION ([ACL Anthology](https://aclanthology.org/2023.acl-long.783)).
- Zhou et al., "Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering", ICLR 2024, arXiv 2309.17249. Bias = mean label score over test batch; zero-shot, inference-only; learnable variant BCL. FROM-PRIOR-KNOWLEDGE; full authors UNVERIFIED.
- Han, Hao, Dong, Sun, Wei, "Prototypical Calibration", ICLR 2023, arXiv 2205.10183. GMM-learned decision boundaries. FROM-PRIOR-KNOWLEDGE.
- Reif and Schwartz, "Beyond Performance: Quantifying and Mitigating Label Bias in LLMs", NAACL 2024, arXiv 2405.02743. 279 tasks, 10 LLMs; size, demos, instruction tuning reduce but do not remove label bias; calibration can hurt; proposes few-shot calibration from demo predictions. VERIFIED-THIS-SESSION ([arXiv](https://arxiv.org/html/2405.02743v1)).
- Min, Lewis, Hajishirzi, Zettlemoyer, "Noisy Channel Language Model Prompting", ACL 2022, arXiv 2108.04106. P(input|label) scoring is more robust; channel prompt tuning handles labels unseen in training. VERIFIED-THIS-SESSION via search ([arXiv](https://arxiv.org/pdf/2108.04106)).
- Cho et al., "Token-based Decision Criteria Are Suboptimal in In-context Learning" (Hidden Calibration), arXiv 2406.16535. Centroid classification on hidden states instead of label-token probabilities. Venue UNVERIFIED.
- Zheng, Zhou, Meng, Zhou, Huang, "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024, arXiv 2309.03882. Option-ID token bias; PriDe debiases via option permutation. FROM-PRIOR-KNOWLEDGE.
- Pezeshkpour and Hruschka, arXiv 2308.11483: option-order sensitivity in MCQ. Venue UNVERIFIED.
- Sclar, Choi, Tsvetkov, Suhr, "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design", ICLR 2024, arXiv 2310.11324. Formatting alone swings accuracy up to about 76 points (LLaMA-2-13B); FormatSpread. Exact number UNVERIFIED.
- Lu et al., "Fantastically Ordered Prompts", ACL 2022, arXiv 2104.08786. Demo order effects. FROM-PRIOR-KNOWLEDGE.
- Verbalizer choice: Gao, Fisch, Chen, LM-BFF, ACL 2021, arXiv 2012.15723; Schick and Schütze, PET, EACL 2021, arXiv 2001.07676; Hu et al., KPT, ACL 2022, arXiv 2108.02035. FROM-PRIOR-KNOWLEDGE.

### Inferences
- PriDe and batch/domain calibration are strong inference-time baselines for chapter 2.
- These methods assume stable label tokens; they do not apply cleanly to per-instance random symbols. Hidden-state or definition-scoring approaches are closer in spirit.

### Gaps
- No calibration method found that conditions on definitions or guarantees renaming invariance.

## Q3. Label descriptions or definitions instead of names

### Takeaway
Across zero-shot classification, NER/IE and task-oriented dialogue, descriptions or guidelines help, and some systems mask or index labels so the model must read descriptions. GoLLIE (class-name masking) and D3ST (description-driven indices) are the closest precedents to chapter 1.

### Cited Findings
- Gao, Ghosh, Gimpel, "The Benefits of Label-Description Training for Zero-Shot Text Classification", EMNLP 2023, arXiv 2305.02239. Fine-tune on label descriptions (terms, dictionary entries, templates). 17 to 19% absolute over zero-shot (EMNLP; arXiv v1 says 15 to 17%); less sensitive to patterns and label-to-token mappings; cross-domain transfer. VERIFIED-THIS-SESSION ([ACL Anthology](https://aclanthology.org/2023.emnlp-main.853)).
- Sainz, García-Ferrero, Agerri, Lopez de Lacalle, Rigau, Agirre, "GoLLIE", ICLR 2024, arXiv 2310.03668. Labels as Python classes, guidelines as docstrings. Regularizers: class order shuffling, class dropout, guideline paraphrasing, candidate sampling, class name masking (e.g., "Person" to LABEL_1). Zero-shot F1: full 55.3; w/o masking 54.6 (p = 0.10, n.s.); w/o dropout 54.0 (p = 0.004); w/o candidates 49.9; w/o all 42.3. Notes ACE05 vs CoNLL03 disagree on pronouns as PERSON. VERIFIED-THIS-SESSION ([arXiv HTML](https://arxiv.org/html/2310.03668v5)).
- Zamai et al., "Show Less, Instruct More" (SLIMER), arXiv 2407.01272 (2024). Definitions plus guidelines for zero-shot NER; comparable to SOTA out of domain with reduced tag set. Venue UNVERIFIED ([arXiv](https://arxiv.org/abs/2407.01272v1)).
- Zhao, Gupta, Cao, Yu, Wang, Lee, Rastogi, Shafran, Wu, "Description-Driven Task-Oriented Dialog Modeling" (D3ST), arXiv 2201.08904 (2022). Schema names invite memorization; uses descriptions plus index-picking. Better DST, data efficiency, zero-shot on MultiWOZ, SGD, SGD-X. That indices are randomly permuted per example is UNVERIFIED this session ([arXiv](https://arxiv.org/abs/2201.08904)).
- Lee, Gupta, Rastogi, Cao, Zhang, Wu, "SGD-X", AAAI 2022, arXiv 2110.06800. Five crowdsourced paraphrased schema variants (names and descriptions); SOTA DST drops 12 to 18% JGA; schema sensitivity metric. VERIFIED-THIS-SESSION ([arXiv](https://arxiv.org/pdf/2110.06800)).
- Coca et al. 2023, arXiv 2303.09905 (paraphrase-ranked augmentation) and arXiv 2309.13448 (grounding with knowledge-seeking turns) improve SGD-X robustness. Venues UNVERIFIED.
- Gupta et al., "Show, Don't Tell", NAACL 2022, arXiv 2204.04327. One example dialogue beats descriptions and is more robust on SGD-X. FROM-PRIOR-KNOWLEDGE.
- Yin, Hay, Roth, EMNLP 2019, arXiv 1909.00161. Entailment zero-shot classification with label names or definitions. FROM-PRIOR-KNOWLEDGE.
- Mishra et al., Natural Instructions, ACL 2022, arXiv 2104.08773; Wang et al., Super-NaturalInstructions, EMNLP 2022, arXiv 2204.07705. Task "Definition" field drives cross-task generalization. FROM-PRIOR-KNOWLEDGE.
- Ye et al., "Guess the Instruction! Flipped Learning", ICLR 2023, arXiv 2210.02969. Strong gains on tasks with novel labels. FROM-PRIOR-KNOWLEDGE.
- Ma et al., "Label Semantics for Few Shot NER", Findings ACL 2022, arXiv 2203.08985. Exploits label names (counterpoint). FROM-PRIOR-KNOWLEDGE.

### Inferences
- GoLLIE's small, non-significant masking gain suggests augmentation alone may be insufficient; an explicit invariance objective may be needed.
- D3ST plus SGD-X are the most complete precedent pair and sit close to speech intent/slot tasks.

### Gaps
- D3ST random-permutation detail not confirmed.
- No speech-domain definition-grounded classification paper found.

## Q4. Explicit invariance to label renaming, and consistency measurement

### Takeaway
Renaming invariance is approached via augmentation (symbol tuning, GoLLIE masking, schema paraphrasing), inference-time permutation debiasing (PriDe), and new 2026 architectural/theoretical work. Consistency is measured in SGD-X and implicitly by verbalizer manipulation. No paper was found that trains a definition-conditioned LLM classifier with an explicit consistency loss across label renamings.

### Cited Findings
- Işık and Li, "Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning", ICML 2026 poster, arXiv 2601.23169. Provably renaming-invariant via parallel embedding streams and aggregated attention; targets logic/variable renaming. VERIFIED-THIS-SESSION ([arXiv](https://arxiv.org/abs/2601.23169)).
- Guan and Bradic, "When Symbol Names Should Not Matter: A Logistic Theory of Fresh-Symbol Classification", arXiv 2605.07120 (May 2026). Kernel logistic theory; token-overlap collision graph perturbs ideal template classifier; margin-transfer guarantees; no new loss. VERIFIED-THIS-SESSION ([arXiv](https://arxiv.org/abs/2605.07120)).
- SGD-X schema sensitivity metric (above). VERIFIED-THIS-SESSION.
- PriDe (ICLR 2024): permutation-based option-ID debiasing. FROM-PRIOR-KNOWLEDGE.
- Symbol tuning, GoLLIE masking, verbalizer manipulation, label-description training (above). VERIFIED-THIS-SESSION.
- Order-invariance analogues: PEARL, arXiv 2502.14628 (DRO over demo permutations); InvICL, arXiv 2505.04994. Both reportedly ICLR 2025; authors and venues UNVERIFIED.

### Inferences
- Candidate chapter 2 objective: consistency loss (symmetric KL / JS) between predictions for the same input under different renamings with fixed definitions, possibly worst-case (PEARL-style). Novelty claim rests on absence of evidence from a limited search.
- Evaluation can borrow SGD-X schema sensitivity and the natural/neutral/unnatural verbalizer split.
- Guan and Bradic predict overlap between training and test symbols hurts; use a large disjoint symbol pool.

### Gaps
- No paper found reporting prediction-agreement rates across label renamings for LLM classifiers with definitions in the prompt.
- No speech LLM work on renaming invariance found.
