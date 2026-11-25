# Task generalization, ICL, and instruction/label robustness in speech and audio LLMs

Scope note: compiled 2026-11-21 from web search plus abstract/HTML fetches. "Verified" means the arXiv page or a primary listing was opened or surfaced in search with matching title and authors during this session. Items marked [UNVERIFIED] come from background knowledge and were not checked in this session; confirm the ID and venue before citing. No em-dashes are used.

## Q1. Benchmarks for instruction following and unseen tasks in speech LLMs, and reported sensitivity to prompt wording or label names

### Takeaway
Benchmarks consistently show that speech LLMs do well on seen tasks and poorly on unseen ones, and a 2025 wave of benchmarks (ISA-Bench, Speech-IFEval, MCQ robustness studies) shows large sensitivity to instruction wording, output format and answer-option surface form. None of the benchmarks found systematically varies the label surface form (original vs symbol vs permuted names) of classification tasks while holding the definitions fixed.

### Cited Findings
- Dynamic-SUPERB (Phase 1), Huang et al., ICASSP 2024, arXiv 2309.09510: first instruction-tuning benchmark for universal speech models; baselines did reasonably on seen tasks but struggled on unseen ones, which the authors attribute to models not actually performing tasks from the semantic content of instructions; [arXiv 2309.09510](https://arxiv.org/pdf/2309.09510)
- Dynamic-SUPERB Phase-2, Huang et al., ICLR 2025, arXiv 2411.05361: 180 tasks (125 community-contributed), extends from classification to regression and sequence generation over speech, music and environmental audio; no model is good across the board (e.g., SALMONN-13B strong on English ASR, WavLLM on emotion), models do well only on a narrow range of tasks; [arXiv 2411.05361](https://arxiv.org/abs/2411.05361); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/32056)
- ISA-Bench, Li, Huang, Qiu, Guo, Wang, Li, Peng, Ma, Chen, Yu (SJTU), arXiv 2510.23558, submitted to ICASSP 2026: varies instruction description, output format and task composition; even state-of-the-art LALMs are "highly sensitive" to phrasing, which degrades basic audio understanding. Fine-tuning Qwen2-Audio on complex instruction variants improved instruction following but caused catastrophic forgetting of previously mastered tasks; [arXiv 2510.23558](https://arxiv.org/abs/2510.23558)
- Speech-IFEval, Lu, Kuan, Lee (NTU), Interspeech 2025, arXiv 2505.19037: separates instruction following from perception and compares each speech LM to its text-only base LLM; most speech LMs trail their text bases badly on basic instructions and are sensitive to small prompt changes; [arXiv 2505.19037](https://arxiv.org/abs/2505.19037?context=cs)
- Robustness assessment of LALMs in multiple-choice evaluation, arXiv 2510.04584: text-only LLMs exceed chance on MMAU and MMSU (gemma-3-27B-it reaches 48.3% on MMAU with no audio, 22.6 points above random); the paper studies MCQ robustness (option-level perturbations); [arXiv 2510.04584](https://arxiv.org/html/2510.04584)
- MMAU-Pro, arXiv 2508.13992: built with stronger distractors so that text-only models drop to 16 to 30%, i.e., designed to resist language-prior shortcuts present in MMAU; [arXiv 2508.13992](https://arxiv.org/html/2508.13992v1)
- MMAU, Sakshi et al., ICLR 2025 (arXiv 2410.19168 [UNVERIFIED id]): massive multi-task audio understanding MCQ benchmark; [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d36f208919582785db965fe648b9fe59-Paper-Conference.pdf)
- AHAMask, Guo, Li, Wang, Li, Wang, Chen, Yu, AAAI 2026, arXiv 2509.01787: states that LALMs "suffer from prompt sensitivity, where different instructions of the same intention can yield drastically different outcomes"; proposes learning attention-head masks (parameters equal to number of heads) that trigger a task without any instruction and match or beat instructions; [arXiv 2509.01787](https://arxiv.org/abs/2509.01787). A search-result snippet (not confirmed in the abstract) says SALMONN and Qwen2-Audio change behavior under same-intent rephrasings, e.g., ASR WER rising to about 12% with an all-caps instruction; [arXiv HTML](https://arxiv.org/html/2509.01787)
- Survey on Speech LLMs for Understanding, arXiv 2410.18908: lists instruction sensitivity as a key challenge and probes Qwen2-Audio, Qwen2.5-Omni and SALMONN across ASR, translation, emotion and gender under punctuation, casing, phrasing and format variants (per search snippet); [arXiv 2410.18908](https://arxiv.org/html/2410.18908v5)
- Other benchmarks [UNVERIFIED ids and venues, not checked this session]: AIR-Bench (Yang et al., ACL 2024, arXiv 2402.07729), AudioBench (Wang et al., NAACL 2025, arXiv 2406.16020), VoiceBench (Chen et al., arXiv 2410.17196), SAKURA multi-hop speech/audio reasoning (Yang et al., Interspeech 2025, arXiv 2505.15406), SpeechInstructBench (arXiv 2503.02769). DeSTA2.5-Audio reports results on Dynamic-SUPERB, MMAU, SAKURA, Speech-IFEval and VoiceBench, which confirms these are the standard suite; [arXiv 2507.02768](https://arxiv.org/html/2507.02768v2)

### Inferences
- The field's sensitivity studies perturb instruction wording, format and MCQ option order, but not class-label names inside a task definition. "Label surface form" robustness (symbol vs permuted vs original) is a distinct axis that is not covered by ISA-Bench or Speech-IFEval.
- ISA-Bench's finding that instruction-variant fine-tuning causes forgetting is directly relevant to Chapter 2: any invariance objective must be checked for forgetting on seen tasks.
- MMAU's text-only above-chance scores mean label names carry exploitable priors; a permuted-label regime is a natural control for this.

### Gaps
- Could not retrieve per-model numbers for Phase-2 seen vs unseen splits, or whether Phase-2 reports instruction-paraphrase variance.
- Did not verify whether AIR-Bench, AudioBench, VoiceBench or SAKURA report sensitivity to label names (I believe they do not, but unverified).

## Q2. Speech in-context learning, symbols, label words, flipped labels

### Takeaway
Speech ICL work mostly targets ASR/translation (SALM, COSMIC, Whisper SICL) or task recognition. The only speech-LLM work found that manipulates label surface form during training is Agrawal and Ganapathy (Interspeech 2025), which is the author's own prior work. AudioICL-Bench (SLT 2026) is the first external benchmark explicitly testing task learning with per-episode resampled input-label rules.

### Cited Findings
- Hsu, Chang, Li, Lee, "Exploring In-Context Learning of Textless Speech Language Model for Speech Classification Tasks", arXiv 2310.12477 (Interspeech 2024 [UNVERIFIED venue]): base textless speech LM (GSLM) shows no ICL; a warmup training stage gives it demonstration-learning ability on unseen classification tasks; framed as the first speech LM doing unseen classification via ICL; [arXiv 2310.12477](https://arxiv.org/abs/2310.12477). Note: Agrawal and Ganapathy cite this as "Chang et al. (2024)"; check author order before citing.
- SALM, Chen et al. (NVIDIA), ICASSP 2024 [UNVERIFIED venue], arXiv 2310.09424: speech-augmented LM with multitask and in-context learning for ASR and AST (keyword boosting via ICL); not about classification labels; [arXiv 2310.09424](https://arxiv.org/html/2310.09424)
- Wang et al., "Can Whisper Perform Speech-Based In-Context Learning?", ICASSP 2024, arXiv 2309.07081: test-time SICL for Chinese dialect ASR with few labelled examples and no gradient updates; WER consistently reduced; [arXiv 2309.07081](https://arxiv.org/pdf/2309.07081); [IEEE Xplore](https://ieeexplore.ieee.org/document/10446502)
- COSMIC, Pan et al., arXiv 2311.02248: data-efficient instruction tuning (under 30M trainable params, 450 h English) yielding speech ICL; up to 33.18 BLEU 0-shot EN-to-X speech translation; [arXiv 2311.02248](https://arxiv.org/html/2311.02248v2)
- Roll et al., EMNLP 2025, arXiv 2505.14887: ICL in Phi-4 Multimodal for speaker/accent adaptation in ASR; about 12 example utterances (about 50 s) lower WER; [arXiv 2505.14887](https://arxiv.org/html/2505.14887v1); [ACL Anthology](https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.219.pdf)
- Audio Flamingo, Kong et al., ICML 2024 [UNVERIFIED venue], arXiv 2402.01831: retrieval-augmented ICL; on unseen labels, few-shot improves classification accuracy by over 10% on average vs zero-shot; [arXiv 2402.01831](https://arxiv.org/html/2402.01831v3)
- UniAudio 1.5 / LLM-Codec, NeurIPS 2024, arXiv 2406.10056: frozen LLM does few-shot audio tasks via a codec aligned to the LLM vocabulary; performance below specialist models; [arXiv 2406.10056](https://arxiv.org/pdf/2406.10056). UniAudio 2.0, arXiv 2602.04683, evaluates few-shot sound/emotion classification as unseen-task generalization; [arXiv 2602.04683](https://arxiv.org/pdf/2602.04683)
- **Agrawal and Ganapathy, "Spoken Language Understanding on Unseen Tasks With In-Context Learning", Interspeech 2025, pp. 4103-4107, arXiv 2505.07731 (author's own prior work):** SALMONN-13B fine-tuned (encoder, Q-Former, LoRA) on SLUE-VoxCeleb sentiment or SLUE-HVB dialog acts, evaluated cross-task (incl. SLUE-VoxPopuli NER). Class definitions in the text prompt; 0 to 5 text-transcript demonstrations retrieved by BGE similarity. Three regimes: Regular FT, Symbol FT (labels replaced by symbols like alpha/beta, after Wei et al.), and Random Label FT (class definitions randomly permuted per mini-batch, consistent within an example). Random Label FT gave the best mismatched results in most settings: vs Regular FT, averaged over few-shot, +10.1% VoxCeleb, +95.3% HVB, +64.3% VoxPopuli (relative); [arXiv 2505.07731](https://arxiv.org/abs/2505.07731); [HTML](https://arxiv.org/html/2505.07731v1). The HTML has apparent typos ("33 SLU tasks", "33-class" VoxCeleb) worth fixing in any thesis citation.
- **AudioICL-Bench, Chen, Lin, Chang, Lu, Lee (NTU), SLT 2026, arXiv 2609.11252 (2026-09-10):** separates Task Recognition (demonstrations cue existing skills) from Task Learning (new input-label mapping must be inferred). Nine tasks; per-episode rules resampled so priors cannot solve them; at k=0 models are near or below chance by design. Across five LALMs, the best bind arbitrary sounds to new labels when perception is easy but fail on temporal measurement and multi-rule composition; [arXiv 2609.11252](https://arxiv.org/abs/2609.11252); [HTML](https://arxiv.org/html/2609.11252)
- Text-side anchors: Symbol tuning, Wei et al., EMNLP 2023, arXiv 2305.08298: fine-tuning with labels replaced by arbitrary symbols forces reliance on input-label mappings and partially restores flipped-label following lost in instruction tuning; [arXiv 2305.08298](https://arxiv.org/html/2305.08298v2)
- Search for flipped-label or symbol-tuning studies applied to Qwen2-Audio, Audio Flamingo 3, SALMONN or other audio LLMs returned no hits other than 2505.07731; [search hit summary pointing to 2505.07731](https://lacuna.tiptreesystems.com/work/spoken-language-understanding-on-unseen-tasks-with-in-context-learning/wrk_405b9f1370f1962eb9d4ade28103172e)

### Inferences
- The label-regime idea (symbols, permuted labels) is, as far as found, only in the author's own Interspeech 2025 paper within speech LLMs. Chapter 1 (DG-ICL) extends this to Qwen2-Audio and Audio Flamingo 3; Chapter 2's invariance objective has no direct speech precedent.
- AudioICL-Bench is the closest external evaluation and could serve as an additional unseen-task testbed for Chapter 2, or must at least be cited as concurrent work on arbitrary label binding.

### Gaps
- MoWE-Audio (believed: Zhang et al., A*STAR, arXiv 2409.06635, mixture of weak encoders, ICASSP 2025) could not be verified this session; it is an encoder-fusion architecture, not an ICL/label method, so low relevance.
- "Meta-Whisper" (Hsu et al. 2024, speech meta-ICL for low-resource ASR) is cited by 2505.07731 but was not independently verified.
- Did not find which five LALMs AudioICL-Bench evaluates or how its labels are constructed (symbols vs words).

## Q3. Robustness of audio LLMs to instruction paraphrase and prompt format, and training fixes

### Takeaway
Task over-fitting to training instructions is a documented failure since SALMONN (2023). Fixes split into (a) diversifying or regularizing training (activation tuning, instruction-variant augmentation, self-generated targets in DeSTA) and (b) removing the instruction channel (AHAMask). None optimize explicitly for prediction consistency across paraphrases or label renamings.

### Cited Findings
- SALMONN, Tang et al., ICLR 2024, arXiv 2310.13289: defines "task over-fitting": instruction-tuned multimodal LLMs get confined to the trained task types (mainly ASR and captioning) and lose cross-modal emergent abilities. Fix: few-shot "activation tuning" stage that restores emergent abilities with little loss on trained tasks (ASR WER roughly 2.1/4.9/9.1 to 2.1/4.9/10.0); long, diverse responses (story-telling, long QA) activate abilities while long ASR labels or text-only fine-tuning do not; [arXiv 2310.13289](https://arxiv.org/html/2310.13289v2)
- ISA-Bench (above): instruction-variant fine-tuning of Qwen2-Audio improves compliance but causes catastrophic forgetting; [arXiv 2510.23558](https://arxiv.org/abs/2510.23558)
- DeSTA2.5-Audio, arXiv 2507.02768 (NTU / Lee group): self-generated cross-modal alignment (the backbone LLM writes its own training targets) to preserve language ability and avoid forgetting; SOTA or competitive on Dynamic-SUPERB, MMAU, SAKURA, Speech-IFEval, VoiceBench; [arXiv 2507.02768](https://arxiv.org/html/2507.02768v2). Predecessors [UNVERIFIED ids]: DeSTA (Lu et al., Interspeech 2024, arXiv 2406.18871) and DeSTA2 (Lu et al., ICASSP 2025, arXiv 2409.20007), descriptive speech-text alignment.
- AHAMask (above): bypasses instructions entirely via learned head masks, motivated by prompt sensitivity; [arXiv 2509.01787](https://arxiv.org/abs/2509.01787)
- Speech-IFEval (above): quantifies forgetting relative to the text base LLM; [arXiv 2505.19037](https://arxiv.org/abs/2505.19037?context=cs)

### Inferences
- The consistent observation that robustness training trades off against forgetting suggests Chapter 2 should report seen-task accuracy and a Speech-IFEval-style forgetting measure alongside invariance metrics.
- AHAMask is a conceptual counterpoint: it treats the instruction as unreliable and removes it, whereas Chapter 2 wants the definition to be the sole carrier of task semantics. Worth contrasting explicitly.

### Gaps
- Did not find a speech paper using consistency or KL losses across paraphrased instructions (the text-LLM analogue of prompt-consistency regularization). "Prompt-agnostic Language Models" (arXiv 2610.37683) surfaced in search but was not opened and its modality is unknown [UNVERIFIED].

## Q4. Audio language priors and hallucination: answering without listening

### Takeaway
There is strong evidence that LALMs lean on text priors: text-only LLMs beat chance on MMAU, and models retain 60 to 72% of full accuracy without audio. Mitigations are mainly audio-contrastive decoding (AAD and successors) and DPO, with model-dependent gains; notably a 2026 audit reports near-zero per-class language prior for Audio Flamingo 3 but a sizable one for Qwen2-Audio.

### Cited Findings
- Kuan, Huang, Lee, "Understanding Sounds, Missing the Questions", arXiv 2406.08402 (Interspeech 2024 [UNVERIFIED venue]): first object-hallucination study for LALMs; captioning is fine but yes/no presence questions are poor; prompt phrasing (asking whether the model "noticed" a sound) changes F1 substantially; [arXiv 2406.08402](https://arxiv.org/pdf/2406.08402)
- Hsu, Lu, Chiang, Lee, Audio-Aware Decoding (AAD), arXiv 2506.07233: contrasts next-token logits with vs without audio; F1 +0.046 to +0.428 across three LALMs on object hallucination, plus gains on general audio QA; [arXiv 2506.07233](https://arxiv.org/abs/2506.07233v2)
- "All That Glitters Is Not Audio", arXiv 2604.24401: removes audio entirely (argues silence is a confounded proxy); models retain about 60 to 72% of full-audio accuracy without audio; [arXiv 2604.24401](https://arxiv.org/pdf/2604.24401)
- Audio-contribution-aware post-training, arXiv 2509.21060: 30 s silence substitution gives about 49.8% on MMAU-test-mini vs 25.5% random; [arXiv 2509.21060](https://arxiv.org/pdf/2509.21060)
- "Does Location Matter? A Black-Box Causal Audit...", arXiv 2610.13112: on Qwen2-Audio, LangPrior-CD fixes 40.7% of hallucinations but breaks 11.1% of correct claims; on Audio Flamingo 3 no gain because its per-class language prior is near zero; [arXiv 2610.13112](https://arxiv.org/pdf/2610.13112) (per search snippet; full text not opened)
- HalluAudio, arXiv 2604.19300: paired minimal-edit prompts/audio; models default to language priors rather than audio grounding; [arXiv 2604.19300](https://arxiv.org/html/2604.19300v1)
- MISHAP-Bench, arXiv 2609.33893: common-protocol hallucination benchmark comparing AAD, VCD, MTI, DoLa; entropy gating measures uncertainty, not grounding, so confident hallucinations persist; [arXiv 2609.33893](https://arxiv.org/html/2609.33893v1)
- Adaptive Perturbation Selection for Contrastive Audio Decoding, arXiv 2607.00247: task-specific negative branches beat total masking/noise; constraining to yes/no tokens reduces affirmative bias; [arXiv 2607.00247](https://arxiv.org/pdf/2607.00247)
- Multimodal DPO for hallucination reduction, arXiv 2610.04004: reports outperforming contrastive decoding (76.7% with Qwen2 backbone for CD) and notes CD does not fix acoustic representations; [arXiv 2610.04004](https://arxiv.org/html/2610.04004)
- Pay More Attention to Audio, arXiv 2509.18816: attributes errors to cross-modal attention imbalance; [arXiv 2509.18816](https://arxiv.org/pdf/2509.18816) (title only seen)

### Inferences
- Permuted-label and symbol regimes are an under-used diagnostic for language priors: if a model's prediction tracks the label word rather than the definition, it is a text-prior failure analogous to these hallucination findings. Chapter 2 can frame label-surface invariance as a classification-specific form of audio grounding.
- The Qwen2-Audio vs Audio Flamingo 3 asymmetry in language prior (2610.13112) predicts that the two backbones in DG-ICL may differ in how much the original-label regime benefits from priors; worth checking in Chapter 1 data.

### Gaps
- Most 2026 hallucination papers were seen only through search snippets; numbers above should be checked in full text. Venues for 2026 preprints unknown.

## Q5. Closest work to definition-grounded, label-surface-invariant task generalization for speech LLMs (overlap check)

### Takeaway
No external paper found does definition-grounded, label-surface-invariant generalization to unseen speech classification tasks. The closest overlaps are the author's own Interspeech 2025 paper (prior art for the label regimes), AudioICL-Bench (concurrent evaluation of arbitrary label binding in LALMs), and ISA-Bench/AHAMask (instruction sensitivity).

### Cited Findings
- Highest overlap, own work: Agrawal and Ganapathy, Interspeech 2025, arXiv 2505.07731: definitions in prompt, symbol and per-batch permuted-label fine-tuning, unseen-task SLU with SALMONN; [arXiv 2505.07731](https://arxiv.org/html/2505.07731v1)
- High overlap, evaluation side: AudioICL-Bench (SLT 2026, NTU): arbitrary, per-episode resampled label rules to test task learning in LALMs; overlaps with the random/permuted regime as an evaluation idea but uses demonstrations rather than definitions and does not train for invariance; [arXiv 2609.11252](https://arxiv.org/abs/2609.11252)
- Medium overlap: ISA-Bench (instruction-wording sensitivity and fine-tuning fix with forgetting), AHAMask (instruction-free task specification); [arXiv 2510.23558](https://arxiv.org/abs/2510.23558); [arXiv 2509.01787](https://arxiv.org/abs/2509.01787)
- Medium overlap: Dynamic-SUPERB seen vs unseen task gap, attributed to not using instruction semantics; [arXiv 2309.09510](https://arxiv.org/pdf/2309.09510)
- Low overlap: Interspeech 2026 Audio Encoder Capability Challenge trains an LLM decoder on fixed-label classification (KWS, speaker ID, emotion); no label-set change; [arXiv 2603.22728](https://arxiv.org/html/2603.22728)
- A targeted search for "label permutation", "arbitrary labels", "task definitions" with speech LLM returned nothing relevant beyond the above; [search context: SpeechLLM-as-Judges, ACL 2026](https://aclanthology.org/2026.acl-long.349.pdf) and similar were unrelated.

### Inferences
- Novel space for Chapter 2: (1) an explicit invariance objective (e.g., consistency of predictions across original/symbol/permuted renderings of the same definition), (2) robustness to definition paraphrase, (3) evaluation of both on unseen tasks for Qwen2-Audio and Audio Flamingo 3. None of these three are combined anywhere found.
- Must-cite as concurrent/related: AudioICL-Bench, ISA-Bench, AHAMask, Speech-IFEval, plus symbol tuning on the text side.
- Risk: the Lee group at NTU (AudioICL-Bench, Speech-IFEval, DeSTA, AAD) is actively working on adjacent ICL and robustness questions; a training-side follow-up to AudioICL-Bench would be the most likely source of future overlap.

### Gaps
- Searches were capped at about 15 tool calls; OpenReview (ICLR 2026/2027 submissions) and ACL 2026 proceedings were not scanned systematically, so a closely overlapping 2026 paper could still exist.
- Could not confirm whether any work measures consistency of speech-LLM predictions under definition paraphrase specifically (as opposed to instruction paraphrase).
