# Research Approach: Symbol-Based In-Context Learning for Speech LLMs

## Overview

This document describes our research programme on improving in-context learning (ICL) for speech-text large language models. The core hypothesis is that standard LLM pre-training encodes strong priors about class label semantics (e.g. "positive", "neutral"), causing models to rely on those priors rather than genuinely attending to in-context audio demonstrations. We address this through **symbolic label substitution**, progressively refined via preference optimisation.

---

## Background and Motivation

### The Audio Grounding Problem

In multimodal ICL, models can shortcut by exploiting text patterns instead of attending to the actual audio/visual content of demonstrations. Evidence: removing or blanking demonstration audio leaves performance largely unchanged — the model was not using it.

The analogous problem in vision was studied in:

> **SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization**
> Hongrui Jia, Chaoya Jiang, Haiyang Xu, Wei Ye, et al. (Alibaba / Peking University), Nov 2024
> - Paper: https://arxiv.org/abs/2411.11909
> - GitHub: https://github.com/hongruijia/SymDPO

SymDPO replaces text answers in ICL demonstrations with semantically meaningless symbols (e.g. "narrow" → "rhondda"), forcing the model to ground predictions in visual content. It achieves up to +8.2 CIDEr on captioning and +10.8 CIDEr when combined with retrieval-based example selection — with only ~1 hour of DPO training on 10k samples.

### Key Insight Applied to Speech

For speech ICL, the shortcut is **acoustic-to-text pattern matching**: a model may recognise speaker type or sentiment from superficial acoustic statistics without attending to the demonstration context. Symbolic labels break this the same way — if "positive" is replaced by "rhondda", the model must look at the audio demonstrations to learn what "rhondda" means.

---

## Our Approach

### Current Methods (Implemented)

#### 1. Fixed Symbol Adapter (SFT baseline)
Replace class labels with random 4–5 character tokens (e.g. `neutral → tepj`) throughout training. The same mapping is used for all epochs. Forces the model to use in-context examples rather than pre-trained label knowledge.

**Limitation**: SFT does not provide a contrastive signal — it trains the model to predict the correct symbol but does not explicitly discourage wrong-symbol predictions.

#### 2. Dynamic Symbols (`--dynamic_symbols`)
Generate new random symbol sets each epoch (or each instance). Prevents the model from memorising a fixed symbol-to-label mapping and forces generalisation of the ICL mechanism itself.

#### 3. D-SPO — Differentiable Symbolic Preference Optimisation (`--diff_symbol_enabled`)
Learns the optimal symbol-to-label mapping end-to-end via a differentiable slot matrix (Gumbel-Softmax, straight-through estimator).

- Each slot owns a **private non-overlapping pool** of K candidate tokens — no two slots compete for the same token
- Slot→label assignments rotate on a configurable schedule so all slots receive training signal
- At validation, the top-K most confident slots are selected automatically
- Enables **cross-task transfer**: learned slot tokens act as reliable label placeholders on unseen tasks without retraining the router

**Limitation**: D-SPO still uses SFT-style supervision; no explicit preference signal between chosen and rejected symbol assignments.

---

### Planned Extension: SymDPO-Style DPO for Speech ICL

#### Motivation
DPO provides a contrastive signal that SFT lacks: it simultaneously reinforces the correct symbol-audio mapping (chosen) and penalises incorrect mappings (rejected). SymDPO showed that this contrastive signal — applied on top of symbolic demonstrations — significantly outperforms SFT with symbols alone.

#### Core Idea
Construct DPO preference pairs where:
- **Chosen**: model predicts the symbol that correctly maps to the query audio, given symbolic demonstration context
- **Rejected**: model predicts a wrong symbol (from the same label set)

The demonstration answers are replaced with symbols (fixed or dynamic), so the model cannot resolve the correct answer from text semantics alone — it must attend to the audio demonstrations.

#### Context Format: Definitions vs. Examples

| Format | Description | Trade-off |
|---|---|---|
| **Examples** (preferred) | Few-shot demonstrations: `[audio₁, Q₁, sym_A] [audio₂, Q₂, sym_B] → predict` | Harder to shortcut; forces genuine audio grounding |
| **Definitions** | Explicit: `sym_A = speaker with low pitch` | Brittle — model can shortcut via the definition text |

**Decision**: Use example-based context, matching the SymDPO paradigm.

---

## Experimental Roadmap

| Stage | Approach | Key Question |
|---|---|---|
| **1** | SFT + fixed symbols (current) | Does symbol learning work? What is the ceiling? |
| **2** | D-SPO (current) | Does differentiable symbol learning outperform fixed SFT? |
| **3** | DPO + fixed symbols (SymDPO-style) | Does contrastive DPO outperform SFT under same symbolic setup? |
| **4** | DPO + dynamic symbols | Does dynamic symbol DPO outperform fixed symbol DPO? |
| **5** | D-SPO + DPO combined | Do the two approaches compound? |

### Stage 3 is the critical experiment
If DPO with fixed symbols significantly outperforms SFT with fixed symbols, the contrastive signal is doing real work and justifies the subsequent stages. Run Stage 3 before investing in Stages 4–5.

### Runtime Reward Function (Future)
A runtime reward function would move this into online DPO / RLHF territory. Deferred until offline DPO is validated in Stage 3.

---

## Evaluation Protocol

Following the existing validation framework:

| Mode | Description |
|---|---|
| `fixed` | Same symbols used during training — tests symbol mapping quality |
| `fresh` | New symbols never seen in training — tests generalisation of ICL mechanism |
| `original` | Original label names — tests whether symbol training degraded base capability |

Primary metric: `macro_f1_with_invalid` (out-of-vocabulary true labels excluded; invalid predictions counted as wrong).

---

## References

| Resource | Link |
|---|---|
| SymDPO paper | https://arxiv.org/abs/2411.11909 |
| SymDPO GitHub | https://github.com/hongruijia/SymDPO |
| This project | `/home/aneeraj/ICL-speech-text-LLM` |
