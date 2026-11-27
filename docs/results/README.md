# Chapter 2 results

One file per experiment. Plan: [`docs/chapter2_plan.md`](../chapter2_plan.md).
Each file states what was run, the numbers, and what it implies.

| File | Experiment | Key result |
|---|---|---|
| [e0_e2.md](e0_e2.md) | E0 pilots, E1 surrogate mixture, E2 consistency objective, gold-definition diagnostic, label-format study (Aug to 21 Nov) | Each model reads the task best in the label format it was trained on |
| [e3_dg_trap.md](e3_dg_trap.md) | E3 DG-Trap A0-A3, sad-sink analysis, A2 vs Sym on fixed symbol draws (22-25 Nov) | A2 beats Sym with names by 5 points, trails it with symbols by 2.8; a third of that gap is copying |
| [e4_naming_eval.md](e4_naming_eval.md) | RAVDESS paraphrase, label1 test, strict rescore, pre-trained Qwen (26 Nov) | IT collapses under label1; Sym is the only model better with label1 than with names; the paper undercommits |
| [e5_trap_variants.md](e5_trap_variants.md) | A2 + span, symbol, paraphrase, KL, DPO (trained 26 Nov) | Pending test results |
| [artifacts.md](artifacts.md) | Where predictions, logs and subsets live | |
