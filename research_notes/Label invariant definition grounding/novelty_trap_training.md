# Novelty search: trap training (gold flip + trap margin + deciding-word conditional term + near-miss definitions)

Search date: 2026-11-22. Sources are arXiv, ACL Anthology, OpenReview/ICLR, and CVF.
Tags: [V] means the abstract or paper page was checked. [S] means only search snippets were checked. [M] means the entry is from memory and unverified; check it before citing.

Components of our method:
- C1: gold-flip prompt swaps, so the gold's real name sits on a wrong definition (the trap). Also non-gold swaps and no-swap prompts.
- C2: a reference-free margin preference loss at the first token where gold and trap differ, added to NLL.
- C3: a conditional term, p(gold | full defs) > p(gold | gold def with its deciding words removed). Deciding words are the words not shared with the nearest-neighbour definition, found with sentence embeddings.
- C4: random dropout of non-deciding words on every line, so a damaged line is not a cue.
- C5: LLM-generated near-miss definitions used as a hard-negative class.
- C6: speech LLM (Qwen2-Audio + LoRA) evaluated on unseen classification tasks.

## 1. Hard-negative or near-miss label descriptions

| Paper | What it does | Overlap |
|---|---|---|
| Yuksekgonul et al., "When and Why VLMs Behave like Bags-of-Words", ICLR 2023, arXiv 2210.01936 [V] | NegCLIP adds word-order-perturbed hard negative captions and nearest-neighbour images to contrastive training. | Medium, C5 in spirit: hard negative text descriptions. They are captions, not class definitions, and there is no generative classification. |
| Kamath et al., "The Hard Positive Truth about VL Compositionality", 2024, arXiv 2409.17958 [S] | Shows hard-negative finetuning makes models oversensitive, failing on hard positives (meaning-preserving edits). | Low-medium. It motivates C4: invariance to harmless edits so that damage is not a cue. |
| Ghosh et al., CompA / CompA-CLAP, ICLR 2024, arXiv 2310.08753 [V] | Audio-language CLAP trained with GPT-4-generated composition-aware hard negative captions. | Medium for C5 and C6 (audio and LLM-generated hard negatives). Contrastive retrieval with captions, not definitions in a prompt for a speech LLM. |
| Gao, Ghosh, Gimpel, "Benefits of Label-Description Training for Zero-Shot Text Classification", EMNLP 2023, arXiv 2305.02239 [V] | Finetunes on texts that describe each label. | Low. It uses descriptions as training data but has no negatives, swaps, or preference loss. |
| SemSup-XC, arXiv 2301.11309 [S] | Contrastive extreme classification over sampled negative label descriptions. | Low-medium for C5. Negatives are other real classes, not synthetic near-misses, and it is an encoder model. |
| Zhou et al., ZOE zero-shot entity typing with type definitions, EMNLP 2018 (D18-1231) [S]; LITE (NLI-based entity typing with sampled negative types) [S] | Typing by matching against type definitions, plus negative type sampling. | Low. |
| Generalized zero-shot intent detection with hard intents, NAACL Findings 2025 (2025.findings-naacl.52) [S] | Hybrid contrastive loss that samples hard utterances and hard intents. | Low-medium for C5. The hard negatives are real intents, not generated definitions. |
| Wu et al., BLINK, EMNLP 2020 [M] | Entity linking that matches mentions to entity descriptions, with hard negative descriptions mined by a bi-encoder. | Low-medium for C5 (hard-negative descriptions). It is retrieval, not a prompt list. |
| PromptMix, arXiv 2310.14192 [S]; Li et al., "Generating Hard-Negative Out-of-Scope Data", LREC-COLING 2024, arXiv 2403.05640 [S] | An LLM generates near-boundary examples or hard-negative utterances. | Low. These are synthetic inputs, not synthetic class definitions. |
| Discriminative description selection for CLIP, arXiv 2412.11917 [S] | Training-free selection of descriptions that separate similar classes. | Low-medium for C3: the idea of discriminative words versus a confusable class. |
| Sainz et al., GoLLIE, ICLR 2024, arXiv 2310.03668 [V] | Guideline-following IE LLM. During training it uses class name masking (LABEL_1 placeholders "force it to attend and understand the guidelines"), class dropout, guideline paraphrasing, and class order shuffling. | **High in goal**, partly for C1 and C4. It decouples names from definitions by masking names. It does not use adversarial gold-name swaps, a trap, a preference loss, or a deciding-word ablation. |

## 2. Swapped or counterfactual labels plus a penalty on the prior label

| Paper | What it does | Overlap |
|---|---|---|
| Wei et al., Symbol tuning, EMNLP 2023, arXiv 2305.08298 [V] | Finetunes on ICL with labels replaced by arbitrary symbols, which restores the ability to follow flipped labels. | Medium for C1. It removes label semantics, but has no adversarial real-name swap and no preference loss. |
| Jia et al., SymDPO, CVPR 2025, arXiv 2411.11909 [V] | LMM ICL where answers are replaced by symbols. DPO pairs use contextually mismatched symbols as rejected answers. | **High** for C1 and C2: a preference against a mismatched label token. It is vision ICL with demonstrations (not definitions), uses reference-based sequence DPO, and has no conditional term. |
| Bi et al., Context-DPO, Findings ACL 2025, arXiv 2412.15280 [V] | Counterfactual contexts. DPO with the faithful answer as chosen and the stubborn parametric answer as rejected. It analyses logits at the tokens that separate the two. | **High** for C2 (the "trap" is the prior answer). It is QA with knowledge conflicts, not label definitions, and has no conditional term. |
| KnowPO/KaPO, arXiv 2408.03297 [S] | Knowledge-aware DPO for RAG conflicts. | Medium for C2. |
| Li et al., "LLMs with Controllable Working Memory", arXiv 2211.05110 [S] | Trains on contexts that conflict with world knowledge, with NLL only. | Medium for C1 (counterfactual training without a preference loss). |
| Agrawal and Ganapathy, "SLU on Unseen Tasks with In-Context Learning", Interspeech 2025, arXiv 2505.07731 [V] (own prior work) | SALMONN with symbol and random-label finetuning for unseen SLU tasks. | Medium for C1 and C6. It is the baseline lineage: label-agnostic training without definitions, traps, or preference. |
| Mohammadi et al., "Do LLMs Adhere to Label Definitions?", EMNLP 2025, arXiv 2509.02452 [V] | Swapped and corrupted definitions as a probe. | Evaluation only. Motivation. |
| Fonseca and Cohen, "Can LLMs Follow Concept Annotation Guidelines?", ACL 2024, arXiv 2311.08704 [V] | Counterfactual definitions in zero-shot classification. Only models of 70B or more partly follow them. | Evaluation only. Motivation. |
| Mishra et al., "Definitions, Label Names, or Demonstrations?", arXiv 2611.09457, Nov 2026 [V] | Counterfactual audit that reassigns definitions across labels. Standard finetuning strengthens the original mapping. Aligned demonstrations increase adherence. | **Medium-high as motivation and closest recent threat.** It is a direct empirical precursor for C1, but proposes no trap loss. Cite it and compare. |
| He et al., "When Better Codebooks Are Not Enough", arXiv 2606.06781 [V] | Reassigned label-definition mappings make every method fall below 0.20 F1 (prompting and retrieval). | Evaluation only. Motivation. |
| Peng et al., ICCD, ACL 2025, arXiv 2502.13738 [V] | Contrastive decoding with negative (corrupted-mapping) ICL examples. | Low. Inference only. |
| Kung and Peng, ACL 2023, arXiv 2305.11383 [S]; Yin et al., arXiv 2306.01150 [S] | Instruction tuning works even with definitions stripped to label space. | Motivation for C3: models do not read definitions. |

## 3. Conditional preference by removing the decisive part of the context

| Paper | What it does | Overlap |
|---|---|---|
| Wang et al., mDPO, EMNLP 2024, arXiv 2406.11839 [V] | Adds a conditional term that prefers the response under the original image over the response under a degraded image, plus a reward anchor. | **High in form** for C3. It degrades the image (random crop), not text, and the degradation is not targeted at decisive content. |
| Cosgrove et al., audio mDPO for hallucination, arXiv 2610.04004 [V, loss not confirmed] | mDPO with acoustic perturbations (reversal, frequency masking, noise) for audio LLMs. | Medium for C3 and C6. It perturbs the audio, not the label definitions. |
| AHA (counterfactual hard negatives for LALMs), arXiv 2512.24052 [S]; MoD-DPO [S, id unverified]; ACPO [S, id unverified] | DPO for audio LLMs with counterfactual or corrupted-modality negatives. | Low-medium for C6. These target hallucination, not label definitions. |
| Tang et al., SSFO, arXiv 2508.17225 [V] | Self-supervised DPO: chosen is the output with context, rejected is the output without context. | Medium for C3: a with/without-context contrast. It removes the whole context, not the deciding words, and the preference is on outputs, not conditionals. |
| Kim et al., Instructive Decoding, ICLR 2024, arXiv 2311.00233 [V]; Shi et al., Context-Aware Decoding, 2023 [M] | Contrast logits under the true instruction or context against a noisy or absent one. | Medium for C3 in spirit, but inference only. |
| Moon et al., MASKER, AAAI 2021, arXiv 2012.09392 [V] | Keyword-masked reconstruction, plus a regulariser for low confidence when keywords are removed. Random non-keyword dropping is used in an ablation [M on details]. | **Medium-high** for C3 and C4 mechanics: confidence should drop without keywords, with dropout of non-keywords. It applies to the input text in an encoder classifier, not to prompt definitions, and uses entropy rather than a pairwise conditional margin. |
| Self-supervised Preference Optimization, arXiv 2409.17791 [S] | RAKE keywords are removed at graded levels to create preference degrees. | Medium for C3: keyword removal as a preference signal on responses. |
| Yan et al., CoIN (Contrastive Instruction Tuning), Findings ACL 2024, arXiv 2402.11138 [V] | Contrastive hidden-state loss between paraphrased and different instructions. | Low-medium. |
| Evolutionary Contrastive Distillation, arXiv 2410.07513 [S]; IOPO, arXiv 2411.06208 [S]; MIPO [S] | Pairs that differ in instruction-critical tokens, input-side preference pairs, and responses under wrong prompts as negatives. | Medium for C3 (input-side or instruction-conditional preference). The text is not a label definition, and deciding words are not chosen against a nearest neighbour. |
| TAB-PO, arXiv 2603.00025 [S]; SparsePO, arXiv 2410.05102 [S] | Token-level DPO. TAB-PO notes that only tokens from the first divergence onward matter. | Low-medium for C2: the first-differing-token detail is known in analysis. |

## 4. Speech or audio LLMs
No paper was found that trains a speech or audio LLM with label definitions in the prompt using swaps, traps, preference losses, or definition ablation. The nearest are 2505.07731 (own work, symbol and random labels), SICL-AT (arXiv 2601.18904 [S], audio ICL post-training), UniverSLU (arXiv 2310.02973 [S], natural-language instructions for unseen SLU tasks), CompA-CLAP (hard negative captions), and audio mDPO and AHA (hallucination DPO).

## Verdict

**Already published (individually):**
- Decoupling label names from definitions in training: GoLLIE (name masking), symbol tuning, and own 2505.07731.
- Preference against a prior or mismatched label token: SymDPO and Context-DPO. The first-divergent-token observation appears in TAB-PO analysis.
- Conditional preference with a degraded condition: mDPO for images, the audio mDPO variant, and SSFO or Instructive Decoding for whole-context removal.
- Confidence dropping when keywords are removed, with non-keyword dropping: MASKER (on inputs, encoder classifier).
- Hard negative descriptions generated by an LLM: CompA-CLAP and NegCLIP (captions, contrastive encoders).
- Evidence that LLMs anchor on names under reassigned definitions: Mohammadi 2025, Fonseca and Cohen 2024, He 2026, and Mishra Nov 2026 (the latter also shows plain finetuning does not fix it).

**Appears unclaimed (no hit found):**
1. An adversarial gold-name flip inside a definition list that makes the gold's real name a trap on a wrong definition, trained with an explicit margin against that trap token. This differs from SymDPO's random symbols and Context-DPO's factual answers.
2. A mDPO-style conditional term where the degraded condition is the gold definition with its contrastive deciding words removed, chosen against the nearest-neighbour definition. This is targeted text ablation, not image cropping or whole-context removal.
3. Matched dropout of non-deciding words on all lines, which explicitly removes the "damaged line" shortcut that the conditional term would otherwise create.
4. All of this in a speech LLM for unseen classification tasks, with LLM-generated near-miss definitions as hard-negative classes.
The combination C1+C2+C3+C4 (plus C5 and C6) is not found in any single paper.

**Closest 3 papers and positioning:**
1. **SymDPO (CVPR 2025).** It shares the idea of preference against mismatched label tokens. Our position: SymDPO removes semantics (random symbols) and uses demonstrations. We keep real names but make them adversarial (the trap), condition on definitions rather than demonstrations, use a reference-free first-differing-token margin, and add a conditional term that tests whether the definition was read.
2. **mDPO (EMNLP 2024).** It shares the conditional preference form. Our position: we move the condition from the image to the text definitions, and make the degradation targeted (contrastive deciding words against the nearest neighbour) rather than random cropping. We add matched dropout so degradation itself is not a cue, which mDPO does not address.
3. **Context-DPO (Findings ACL 2025)**, with GoLLIE (ICLR 2024) as a close fourth. Context-DPO shares the counterfactual context plus a penalty on the prior answer. GoLLIE shares the goal of making the model read the guidelines rather than the names. Our position: Context-DPO's prior is world knowledge in QA, while ours is label-name semantics in a definition list. GoLLIE masks names, which removes the conflict instead of training against it. We train against the conflict, which matches how failures were measured in Mohammadi 2025 and Mishra 2026.

Also cite and contrast with Mishra et al. 2026 (arXiv 2611.09457) as recent evidence that plain finetuning does not fix definition adherence. It is the strongest motivation, and a reviewer may raise it.

Unverified items to check before citing: BLINK details, Context-Aware Decoding (Shi et al. 2023), MASKER's exact entropy loss and dropout details, the MoD-DPO and ACPO arXiv ids, whether audio mDPO (2610.04004) uses a conditional term, and the TAB-PO and SPO contents.
