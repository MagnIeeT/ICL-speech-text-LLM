# Candidate Datasets for Multi-Task Symbol Tuning (AF3)

Goal: expand training-task **breadth** with diverse **SLU classification** task *types* (not more
emotion sets) and add **non-floor held-out** tasks — including **fully-unseen Indian** data — to test
whether per-instance symbol tuning (dpi) generalizes symbol-following. Staying **zero-shot** (no few-shot).
See memory `project_multitask_symbol_plan.md` and `ref_symbol_tuning_paper.md`.

## Selection rules
1. **Must be UNSEEN by AF3** (else "generalization" is contaminated).
2. **SLU classification** = closed label set, utterance-level (intent / dialogue act / keyword / sentiment / language-ID / speaker-trait). Diverse *types* > many emotion sets.
3. **Held-out tasks must be non-floor**: untrained AF3 should score meaningfully above chance in `original` mode, or the orig→fresh drop is unmeasurable (the voxpopuli-at-floor problem).
4. **Easy setup** = openly downloadable, has `audio` + a categorical label; we convert to the repo's on-disk format (`datasets.save_to_disk` → env-var paths, loaded via `load_from_disk`; see `config/data_config/*_config.py`).

## AF3 cross-check (arXiv 2507.08128, Tables 9/10/11 + AF-Whisper — VERIFIED from appendix)
Full training corpus confirmed from the paper's own tables:
- **Emotion (SEEN — do not reuse):** Emov-DB, JL-Corpus, **TESS**, OMGEmotion, **MSP-Podcast**, **MELD**, **IEMOCAP**. (AF3 is a strong emotion recognizer — 7 emotion sets — so held-out emotion tasks are non-floor but the *emotion skill* is not novel.)
- **Speech / ASR / dialogue (SEEN):** LibriSpeech, Switchboard, GigaSpeech, Common Voice 15, **VoxPopuli**, TEDLIUM, SPGISpeech, Fisher, MultiDialog, **VoxCeleb2**, DailyTalk, Spotify Podcasts, EuroParl.
- **Rest of corpus:** music + audio-event captioning/QA (AudioSet, Clotho, FSD50k, NSynth, WavCaps, MusicCaps, …).
- **CONFIRMED ABSENT (genuinely UNSEEN):** CREMA-D, RAVDESS, RAVDESS-song, ESD, HVB (SLUE/HarperValleyBank), and ALL SLU classification benchmarks — **no intent, no keyword spotting, no dialogue-act, no Indian data anywhere in AF3.** So our current hvb/cremad/ravdess setup is clean, and every candidate below is confirmed unseen.

Availability legend: **EASY** = open `load_dataset(...)` on HF · **MED** = license/registration or manual assembly · **HARD** = scattered / synth / needs building.

---

## Current inventory (verified on-disk, node1, 2026-08-02)
What is actually set up right now. "Present" = data verified on `/home/aneeraj/data` (node1). Configs live in `config/data_config/*_config.py`, registered in `master_config.py`.

### Trainable (unseen + has a TRAIN split) — the training pool
| Dataset | Task type | Labels | Lang | AF3 | Data | Notes |
|---|---|---|---|---|---|---|
| **hvb** | dialogue acts | ~34 | en | UNSEEN ✓ | present | current train task |
| **cremad** | emotion | 6 | en | UNSEEN ✓ | present | current train task |
| **skit_s2i** | intent (banking) | 14 | Indian-en | UNSEEN ✓ | present | **dual-use** (train or held-out) |
| **minds14_en** | intent (banking) | 14 | en | UNSEEN ✓ | present | **dual-use**; has definitions → good train task |
| ravdess_song | emotion | 6 | en | UNSEEN ✓ | present | usually held-out corroborator; has train split if needed |

### Eval-only (held-out probes — never train on these)
| Dataset | Probe role | Labels | Lang | AF3 | Data | Notes |
|---|---|---|---|---|---|---|
| **ravdess_song** | emotion corroborator | 6 | en | UNSEEN ✓ | present | primary orig→fresh corroborator |
| **speech_commands** | capability-retention (keyword) | 10 | en | UNSEEN ✓ | present (val/te) | original/transcription only, no symbols |
| **minds14_fr** | cross-lingual (European) | 14 | fr | UNSEEN ✓ | present | same legend, French audio |
| **minds14_ko** | distant unseen language | 14 | ko | UNSEEN ✓ | present | Korean — AF3-unseen SLU language |

### Parked (AF3 **SEEN** — not valid for generalization claims)
voxceleb (sentiment), voxpopuli (entity/intent), meld_emotion (emotion). Configs + data present, but AF3 trained on all three → excluded from symbol-following generalization; kept only for reference/legacy runs.

### Config exists but **NO DATA** (must rebuild before use)
| Dataset | Task | Why unusable | To fix |
|---|---|---|---|
| **esd** | emotion (5) | paths point to `/home/anmola/...` which is not mounted (login node or node1) | download ESD (English speakers 0012–0020), rebuild to on-disk format, repoint env vars |
| **ravdess** (speech) | emotion | same `/home/anmola` path missing | rebuild or drop; `ravdess_song` is the working RAVDESS |

---

## Bucket A — Training breadth (add to hvb + cremad)
Diverse SLU *types*, unseen, to widen the task distribution (Wei Fig 10: more/diverse tasks cure negative transfer).

| Dataset | Task type | Labels | Lang | AF3 | Avail | Notes |
|---|---|---|---|---|---|---|
| **Google Speech Commands** (`google/speech_commands`) | keyword spotting | 35 words (v2) | en | UNSEEN ✓ | **EASY** | very different task type; fully open; huge, subsample |
| **MInDS-14** (`PolyAI/minds14`) | intent (banking) | 14 intents | 14 langs (no Indian) | UNSEEN ✓ | **EASY** | multilingual breadth in one set; ~600/lang |
| **SLURP** (`slurp`, or GitHub audio) | intent + scenario | 18 scenarios / 46 actions | en | UNSEEN ✓ | **MED** | rich; audio from SLURP GitHub, CC-BY |
| **Fluent Speech Commands** | intent (action/object/loc) | 31 combos | en | UNSEEN ✓ | **MED** | clean/small; **license registration** required |
| **MRDA** | dialogue act | ~5 / ~11 acts | en | UNSEEN ✓ | **MED/HARD** | audio = **ICSI meeting corpus** (unseen ✓); segment long meetings by DA span — effortful |
| ~~**SwDA**~~ | dialogue act | ~42 acts | en | **SEEN ✗** | — | **EXCLUDE**: audio is **Switchboard**, which is in AF3's training corpus → contaminated |

## Bucket B — Held-out eval (non-floor corroborators for the ravdess result)
Never trained on; used only to measure orig→fresh symbol-following drop. Must be non-floor.

| Dataset | Task type | Labels | Lang | AF3 | Avail | Notes |
|---|---|---|---|---|---|---|
| **Google Speech Commands** | keyword spotting | 35 | en | UNSEEN ✓ | **EASY** | model can transcribe words → non-floor; use as held-out if NOT in Bucket A |
| **Fluent Speech Commands** | intent | 31 | en | UNSEEN ✓ | **MED** | untrained AF3 likely non-floor; license gate |
| **Common Voice gender/age** | speaker trait | 2–3 / bins | multi | CV seen (task novel) | **EASY** | cheap orthogonal probe; CV audio seen but the *classification task* is new |
| **ravdess_song** (have) | emotion | 6 | en | UNSEEN ✓ | ready | already the primary corroborator |

## Bucket C — Fully-unseen Indian (the strong generalization probe)
No Indian speech in AF3 → truly novel. Watch the floor risk: verify untrained AF3 is non-floor before trusting as eval.

| Dataset | Task type | Labels | Lang | AF3 | Avail | Notes |
|---|---|---|---|---|---|---|
| **Skit-S2I** (`skit-ai/skit-s2i`) | intent (banking) | 14 intents | Indian-English (en-IN) | UNSEEN ✓ | **EASY** | **best immediate pick**: open, clean, 11 Indian speakers; note it's Indian-*accented English*, not an Indian language |
| **IndicVoices** (`ai4bharat/indicvoices`) | ASR + some NLU | varies | 22 Indian langs | UNSEEN ✓ | **MED** | true Indian languages; **verify it has a usable classification label** (base is ASR/read/extempore) |
| **IndicVoices-R** (`ai4bharat/indicvoices_r`) | TTS/ASR | — | 22 Indian langs | UNSEEN ✓ | — | **not a classification task** (TTS corpus) — exclude unless labels exist |
| **IndicSUPERB** | multiple SLU | varies | 12 Indian langs | UNSEEN ✓ | **MED** | check for a classification subtask (e.g. language-ID / intent) |

---

## Recommended first setups (in order)
1. **Skit-S2I** — Bucket C, EASY, unseen, non-floor (English intent), tiny. Immediate generalization probe + easy loader. *(Do first.)*
2. **Google Speech Commands** — Bucket A/B, EASY, a genuinely different task type (keyword spotting) → real breadth.
3. **MInDS-14** — Bucket A, EASY, multilingual intent → breadth in one set.
4. **SLURP** or **FSC** — Bucket A/B, MED effort, once the easy ones show signal.
5. True **Indian-language** classification (IndicVoices/IndicSUPERB) — MED, after verifying a usable label + non-floor AF3 score.

## Setup checklist per dataset
- [x] Confirm truly unseen vs AF3 appendix — DONE: all candidates verified absent from Tables 9/10/11 + AF-Whisper.
- [ ] Download; build `train/validation/test` with `audio` + single categorical label column.
- [ ] `save_to_disk` to a local path; add `<NAME>_TRAIN/VAL/TEST_PATH` env vars.
- [ ] Add `DatasetType.<NAME>` + a `*_config.py` (copy `cremad_config.py`): `valid_labels`, `prompt_template`, `max_new_tokens`.
- [ ] Register in `master_config.py DATASET_CONFIGS`.
- [ ] Sanity-check untrained AF3 `original`-mode score (non-floor?) before using as held-out.
