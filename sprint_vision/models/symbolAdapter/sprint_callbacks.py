"""
SPRInT Trainer Callbacks for VLM training.

These callbacks are registered with HuggingFace Trainer in llava/train/train.py.
They handle: ED-FT symbol rotation, ICI-style progress bar, epoch header
logging, and per-epoch validation with best-checkpoint tracking.

rank0_print and _sprint_microbatch_loss live in llava/train/train.py and are
imported here at call time (deferred) to avoid a circular import at module load.
"""

import os
import sys
import math
import shutil

import torch
import transformers

from .validation import SPRInTValidationManager


def _rank0_print(*args):
    """Deferred import of rank0_print to avoid circular import with train.py."""
    try:
        from llava.train.train import rank0_print
        rank0_print(*args)
    except Exception:
        print(*args)


class SPRInTSymbolEpochCallback(transformers.TrainerCallback):
    """
    Rotates ED-FT symbols at the start of each training epoch.

    HuggingFace Trainer re-spawns DataLoader workers at the start of each
    epoch when persistent_workers=False (the default).  on_epoch_begin fires
    before the new iterator is created, so workers fork with the updated
    current_epoch and call get_current_symbols() → fresh mappings for that
    epoch.  NUM_WORKERS=0 is set in run_sprint_finetune.sh for ed_ft so
    symbol state is shared in-process rather than across fork boundaries.
    """

    def __init__(self, symbol_manager):
        self.symbol_manager = symbol_manager

    def on_epoch_begin(self, args, state, control, **kwargs):
        # round(), NOT int(): HF's state.epoch is a float accumulator that can sit a
        # hair below the integer at the epoch boundary (e.g. 2.9999), so int() floors
        # epoch 3 → 2, overwriting that history slot and dropping a key. round() maps
        # 2.9999 → 3 so every epoch gets its own distinct symbol set (keys 0..N-1).
        epoch = round(state.epoch) if state.epoch is not None else 0
        self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=True)
        if args.local_rank in (-1, 0):
            _rank0_print(
                f"[SPRInT ED-FT] Epoch {epoch}: rotated symbols → "
                f"{self.symbol_manager.get_current_symbols()}"
            )


class SPRInTProgressCallback(transformers.ProgressCallback):
    """
    Replaces HF's ProgressCallback to exactly match ICI's per-epoch tqdm format:

      Epoch 2/5:   0%|          | 0/100 [loss=0.797097]  ← resets to 0% each epoch
      Epoch 2/5:   1%|          | 1/100 [loss=0.797097]  ← on_step_end / on_substep_end advance
      Epoch 2/5:   1%|          | 1/100 [loss=0.419798]  ← on_log updates loss in-place

    Bar total = raw dataloader batches (samples / per_device_batch_size), NOT optimizer steps.
    Both on_step_end and on_substep_end advance the bar by 1 so every micro-batch is counted,
    matching ICI which always counts individual batch forward passes.
    Bar is RECREATED each epoch. Loss shows inside the bar only.
    """

    def __init__(self, train_dataset_len: int = 0):
        super().__init__()
        self._epoch_counter = 0
        self._steps_per_epoch = None
        self._train_dataset_len = train_dataset_len

    def on_epoch_end(self, args, state, control, **kwargs):
        """Close the bar at epoch end so it doesn't reappear with inflated time later."""
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_counter += 1
        total_epochs = int(args.num_train_epochs)

        if state.is_local_process_zero:
            if self.training_bar is not None:
                self.training_bar.close()
                self.training_bar = None
            if self._steps_per_epoch is None:
                import math
                if self._train_dataset_len > 0:
                    num_processes = max(1, getattr(args, 'world_size', 1))
                    self._steps_per_epoch = math.ceil(
                        self._train_dataset_len / args.per_device_train_batch_size / num_processes
                    )
                else:
                    # Fallback when dataset length unknown
                    self._steps_per_epoch = max(1, state.max_steps // total_epochs) * args.gradient_accumulation_steps
            from tqdm.auto import tqdm as _tqdm
            self.training_bar = _tqdm(
                total=self._steps_per_epoch,
                desc=f"Epoch {self._epoch_counter}/{total_epochs}",
                dynamic_ncols=True,
            )

    def on_step_end(self, args, state, control, **kwargs):
        """Advance bar by 1 for each optimizer step (the last micro-batch in each accum cycle)."""
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.update(1)
        self.current_step = state.global_step

    def on_substep_end(self, args, state, control, **kwargs):
        """Advance bar by 1 for each intermediate micro-batch; refresh loss from the cache."""
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.update(1)
            try:
                from llava.train.train import _sprint_microbatch_loss
                if _sprint_microbatch_loss[0] is not None:
                    self.training_bar.set_postfix({"loss": f"{_sprint_microbatch_loss[0]:.6f}"})
            except Exception:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or self.training_bar is None:
            return
        logs = (logs or {}).copy()
        logs.pop("total_flos", None)
        if "loss" in logs:
            self.training_bar.set_postfix({"loss": f"{logs['loss']:.6f}"})
        elif logs:
            self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        # HF 4.37.2's inherited on_train_end calls training_bar.close() without a None guard.
        # Our on_epoch_end already closes the bar, so training_bar is None here → crash.
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.close()
        self.training_bar = None


class SPRInTEpochLogCallback(transformers.TrainerCallback):
    """Prints epoch header at start of each epoch — mirrors ICI's 'Epoch N/Total' line."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self._epoch_num = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_num += 1
        _rank0_print(f"\n{'='*80}")
        _rank0_print(f"Epoch {self._epoch_num}/{self.total_epochs}")
        _rank0_print(f"{'='*80}")


class SPRInTValidationCallback(transformers.TrainerCallback):
    """
    Thin HF Trainer wrapper — delegates all validation logic to SPRInTValidationManager.

    Responsibilities kept here (HF lifecycle only):
      - on_epoch_begin : reset step-loss accumulator
      - on_log         : accumulate per-step losses for avg-loss computation
      - on_epoch_end   : build mode_symbols dict, call manager, track best epoch
      - on_train_end   : call manager.print_consolidated_summaries()

    All inference, metric computation, and logging live in:
      sprint_vision/models/symbolAdapter/validation.py  (SPRInTValidationManager)
      sprint_vision/utils/evaluation_utils.py           (pure metric functions)
    """

    def __init__(self, model, tokenizer, image_processor, data_args, training_args, symbol_manager):
        self.training_args  = training_args
        self.symbol_manager = symbol_manager
        self._step_losses          = []
        self._epoch_counter        = 0
        self._epoch_history        = []
        self._best_score           = -1.0
        self._best_secondary_score = -1.0   # AUC tiebreaker when primary scores tie
        self._best_epoch           = None
        self._is_new_best          = False
        self._warned_no_val        = False

        # Load dataset config to determine label vocabulary, multi-label flag, and primary metric.
        try:
            _sprint_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            if _sprint_dir not in sys.path:
                sys.path.insert(0, _sprint_dir)
            from config.data_config import get_dataset_config as _get_cfg
            _ds  = getattr(training_args, "sprint_dataset", "colon")
            _cfg = _get_cfg(_ds)
            label_names    = [l.lower() for l in _cfg.label_names]
            is_multi_label = _cfg.is_multi_label
        except Exception as _e:
            _rank0_print(f"[SPRInT Val] Warning: could not load dataset config ({_e}). Using binary fallback.")
            _cfg           = None
            label_names    = ["0", "1"]
            is_multi_label = False

        dataset_name = getattr(training_args, "sprint_dataset", "colon")
        # MedFMC paper save_best criteria (from configs/_base_/datasets/*.py):
        #   chest.py:     metric='mAP',          save_best='auto'
        #   colon.py:     metric='accuracy',      save_best='auto'  (topk=1)
        #   endoscopy.py: metric='AUC_multilabel',save_best='auto'  (= macro_auc in our code)
        # Colon: MedFMC uses accuracy (topk=1 = argmax of softmax logits).
        # In our code two accuracy metrics exist:
        #   modes["original"]["accuracy"]  → text-generation string match (acc_exact, noisy)
        #   auc_map["accuracy_aacc"]       → logit-argmax (l1>l0), same as MedFMC top-1
        # We use "accuracy_aacc" (logit path) as it matches the official metric exactly.
        # AUC is the tiebreaker when two epochs have equal accuracy_aacc.
        if dataset_name == "chest":
            self._primary_metric = "mAP"
        elif dataset_name == "endo":
            self._primary_metric = "macro_auc"       # AUC_multilabel in paper
        else:                                         # colon
            self._primary_metric = "accuracy_aacc"   # logit-argmax == MedFMC top-1
        self._validation_modes = getattr(training_args, "validation_modes", "fixed,original,fresh")

        self._val_manager = SPRInTValidationManager(
            model            = model,
            tokenizer        = tokenizer,
            image_processor  = image_processor,
            data_args        = data_args,
            symbol_manager   = symbol_manager,
            label_names      = label_names,
            is_multi_label   = is_multi_label,
            dataset_name     = dataset_name,
            primary_metric   = self._primary_metric,
            validation_modes = self._validation_modes,
            max_val_samples  = getattr(training_args, "max_val_samples", 100),
            eval_data_path   = getattr(training_args, "eval_data_path", None),
            print_fn         = _rank0_print,
            cfg              = _cfg,
        )

        # ── Cross-task validation monitors (Change 2) ─────────────────────────
        # ALSO validate on the OTHER MedFMC datasets each epoch (read-only).
        # best-epoch / checkpoint-best stay driven by the TRAINED dataset ONLY.
        # Default ON; disable with SPRINT_CROSS_TASK_VAL=false. A dataset whose
        # val JSON is absent is skipped (logged), never crashes training.
        self._monitor_managers = {}
        self._monitor_history  = {}
        _cross_on = os.environ.get("SPRINT_CROSS_TASK_VAL", "true").strip().lower() in ("1", "true", "yes")
        _primary_eval = getattr(training_args, "eval_data_path", None)
        if _cross_on and _primary_eval:
            try:
                from config.data_config import get_dataset_config as _get_cfg2
            except Exception as _e:
                _get_cfg2 = None
                _rank0_print(f"[SPRInT Monitor] config import failed ({_e}); cross-task validation disabled.")
            for _d in [x for x in ("colon", "chest", "endo") if x != dataset_name]:
                if _get_cfg2 is None:
                    break
                try:
                    _dcfg = _get_cfg2(_d)
                except Exception as _e:
                    _rank0_print(f"[SPRInT Monitor:{_d}] config load failed ({_e}); skipping.")
                    continue
                # Sibling val JSON: same dir + shot/exp, dataset token swapped.
                _bn    = os.path.basename(_primary_eval).replace(f"{dataset_name}_", f"{_d}_", 1)
                _dpath = os.path.join(os.path.dirname(_primary_eval), _bn)
                _dprim = "mAP" if _d == "chest" else ("macro_auc" if _d == "endo" else "accuracy_aacc")
                self._monitor_managers[_d] = SPRInTValidationManager(
                    model            = model,
                    tokenizer        = tokenizer,
                    image_processor  = image_processor,
                    data_args        = data_args,
                    symbol_manager   = None,          # cross-task → original labels only
                    label_names      = [l.lower() for l in _dcfg.label_names],
                    is_multi_label   = _dcfg.is_multi_label,
                    dataset_name     = _d,
                    primary_metric   = _dprim,
                    validation_modes = "original",    # no symbol modes cross-task
                    max_val_samples  = getattr(training_args, "max_val_samples", 100),
                    eval_data_path   = _dpath,
                    print_fn         = _rank0_print,
                    cfg              = _dcfg,
                )
                self._monitor_history[_d] = []
                _rank0_print(f"[SPRInT Monitor:{_d}] registered (read-only) — val JSON: {_dpath}")

    # ── Loss tracking ─────────────────────────────────────────────────────────

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._step_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._step_losses.append(logs["loss"])

    # ── Per-epoch validation ──────────────────────────────────────────────────

    def on_epoch_end(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        if not self._val_manager._load_val_data():
            if not self._warned_no_val:
                self._warned_no_val = True
                _ep = getattr(self.training_args, "eval_data_path", None)
                _rank0_print(
                    f"\n{'!'*80}\n"
                    f"[SPRInT Val] ⚠️  VALIDATION SKIPPED — no eval data found.\n"
                    f"             eval_data_path = {_ep}\n"
                    f"             The file is missing/empty, so NO metrics and NO best-epoch\n"
                    f"             summary will be produced. Generate the val JSON (run\n"
                    f"             medfmc_to_llava.py --shot N --exp K) or set EVAL_DATA_PATH\n"
                    f"             to an existing JSON (e.g. {{task}}_test.json).\n"
                    f"{'!'*80}\n"
                )
            return

        self._epoch_counter += 1
        epoch        = self._epoch_counter
        total_epochs = int(self.training_args.num_train_epochs)
        strategy     = getattr(args, "sprint_strategy", "regular")

        # Build mode → symbols dict filtered to requested modes
        _all_mode_syms = {"original": {}}
        if self.symbol_manager is not None:
            _all_mode_syms["fixed"] = self.symbol_manager.get_current_symbols()
            _all_mode_syms["fresh"] = self.symbol_manager._generate_symbol_mappings()
        requested    = [m.strip() for m in self._validation_modes.split(",")]
        # Preserve the order the user wrote in VALIDATION_MODES (not _all_mode_syms's
        # fixed original/fixed/fresh insertion order) -- the first mode listed is
        # treated as "primary" for best-epoch selection downstream.
        mode_symbols = {m: _all_mode_syms[m] for m in requested if m in _all_mode_syms}
        if not mode_symbols:
            mode_symbols = {"original": {}}

        entry = self._val_manager.run_comprehensive_validation(
            epoch           = epoch,
            total_epochs    = total_epochs,
            mode_symbols    = mode_symbols,
            strategy        = strategy,
            output_dir      = args.output_dir,
            step_losses     = self._step_losses,
            compute_auc_map = getattr(self.training_args, "compute_val_auc_map", True),
        )
        self._epoch_history.append(entry)

        # Best-epoch tracking — metric per MedFMC paper save_best criteria
        auc_map = entry.get("auc_map", {})
        if self._primary_metric == "accuracy_aacc":
            # Logit-argmax accuracy from auc_map — matches MedFMC top-1 (argmax softmax).
            # Falls back to text-gen accuracy only when auc_map was not computed.
            combined_score   = (auc_map.get("accuracy_aacc") or
                                 entry["modes"].get("original", {}).get("accuracy", 0.0))
            best_metric_name = "acc_aACC"
        elif self._primary_metric == "accuracy":
            combined_score   = entry["modes"].get("original", {}).get("accuracy", 0.0)
            best_metric_name = "accuracy"
        elif auc_map:
            combined_score   = auc_map.get(self._primary_metric,
                               auc_map.get("macro_auc", auc_map.get("AUC", 0.0)))
            best_metric_name = self._primary_metric
        else:
            combined_score   = entry["modes"].get("original", {}).get("macro_f1", 0.0)
            best_metric_name = "macro_f1"

        # AUC tiebreaker: when primary scores are exactly equal (colon accuracy),
        # the epoch with higher AUC wins.
        current_auc = auc_map.get("AUC", auc_map.get("macro_auc", 0.0)) if auc_map else 0.0

        if combined_score > self._best_score:
            self._best_score           = combined_score
            self._best_secondary_score = current_auc
            self._best_epoch           = epoch
            self._is_new_best          = True
            _rank0_print(
                f"★  New best: epoch {epoch}/{total_epochs}  {best_metric_name}={combined_score:.4f}"
            )
        elif (combined_score == self._best_score
              and self._primary_metric in ("accuracy", "accuracy_aacc")
              and current_auc > self._best_secondary_score):
            # Tie on accuracy — AUC tiebreaker
            self._best_secondary_score = current_auc
            self._best_epoch           = epoch
            self._is_new_best          = True
            _rank0_print(
                f"★  New best (AUC tiebreaker): epoch {epoch}/{total_epochs}  "
                f"{best_metric_name}={combined_score:.4f}  AUC={current_auc:.4f}"
            )
        else:
            self._is_new_best = False
            _rank0_print(
                f"   No improvement  (best so far: epoch {self._best_epoch}, "
                f"{best_metric_name}={self._best_score:.4f})"
            )

        # ── Cross-task monitors (read-only; NEVER affect best epoch) ──────────
        for _d, _mgr in self._monitor_managers.items():
            try:
                if not _mgr._load_val_data():
                    _rank0_print(
                        f"[SPRInT Monitor:{_d}] ⚠️  SKIPPED — no val data at "
                        f"{_mgr._eval_data_path}.  Generate it with: "
                        f"medfmc_to_llava.py --tasks {_d} --shot <N> --exp <K>"
                    )
                    continue
                _rank0_print(
                    f"\n{'#'*80}\n[CROSS-TASK MONITOR] dataset={_d}  epoch={epoch}"
                    f"  (read-only — does NOT affect best epoch / checkpoint-best)\n{'#'*80}"
                )
                _m_entry = _mgr.run_comprehensive_validation(
                    epoch           = epoch,
                    total_epochs    = total_epochs,
                    mode_symbols    = {"original": {}},
                    strategy        = f"monitor_{_d}",
                    output_dir      = args.output_dir,
                    step_losses     = [],
                    compute_auc_map = getattr(self.training_args, "compute_val_auc_map", True),
                )
                self._monitor_history[_d].append(_m_entry)
            except Exception as _e:
                _rank0_print(f"[SPRInT Monitor:{_d}] ERROR (skipped; training unaffected): {_e}")

    # ── Best-checkpoint copy ──────────────────────────────────────────────────
    # on_save fires AFTER HF Trainer writes checkpoint-{step} to disk.
    # on_epoch_end (above) sets _is_new_best; we copy here where the file exists.

    def on_save(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        latest_ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        # HF Trainer with LoRA does not write config.json OR non-LoRA trainable
        # weights (e.g. mm_projector, tuned via its own --mm_projector_lr) into
        # epoch checkpoints -- PeftModel.save_pretrained() only serializes the
        # LoRA adapter itself. Write both now, from the model's CURRENT state --
        # on_save fires immediately after THIS epoch's weights were written and
        # before any further training happens, so "current state" == this
        # epoch's state. This must happen before the checkpoint-best copy below
        # so that copy picks up this epoch's own non_lora_trainables.bin, not a
        # stale/missing one. (Previously non_lora_trainables.bin was only ever
        # written once, post-training, from the FINAL epoch's live weights, then
        # fanned out to every checkpoint dir including checkpoint-best --
        # silently overwriting every historical checkpoint's projector weights
        # with the final epoch's. Confirmed via md5sum: every checkpoint-N and
        # checkpoint-best shared one identical non_lora_trainables.bin hash.)
        if os.path.isdir(latest_ckpt):
            model = kwargs.get("model")
            if model is not None:
                model.config.save_pretrained(latest_ckpt)
                if getattr(self.training_args, "lora_enable", False):
                    from llava.train.train import get_peft_state_non_lora_maybe_zero_3
                    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                        model.named_parameters()
                    )
                    if non_lora_state_dict:
                        torch.save(
                            non_lora_state_dict,
                            os.path.join(latest_ckpt, "non_lora_trainables.bin"),
                        )

        if not self._is_new_best:
            return
        self._is_new_best = False
        best_ckpt = os.path.join(args.output_dir, "checkpoint-best")
        if not os.path.isdir(latest_ckpt):
            _rank0_print(
                f"[SPRInT] WARNING: checkpoint-{state.global_step} not found on disk; "
                f"check  point-best not updated. Ensure --save_strategy epoch is set."
            )
            return
        if os.path.exists(best_ckpt):
            shutil.rmtree(best_ckpt)
        shutil.copytree(latest_ckpt, best_ckpt)  # config.json + non_lora_trainables.bin already written above
        _rank0_print(
            f"[SPRInT] checkpoint-best updated → checkpoint-{state.global_step}"
            f"  (epoch {self._best_epoch}, score={self._best_score:.4f})"
        )

    # ── End-of-training summaries ─────────────────────────────────────────────

    def on_train_end(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        if not self._epoch_history:
            return
        self._val_manager.print_consolidated_summaries(
            epoch_history  = self._epoch_history,
            primary_metric = self._primary_metric,
            best_epoch     = self._best_epoch,
            best_score     = self._best_score,
        )
        self._print_cross_dataset_summary()

    def _print_cross_dataset_summary(self):
        """End-of-training table: ALL metrics (macro_f1, exact-acc, aACC, mAP, AUC)
        for the trained dataset + every cross-task monitor, per epoch. The best
        epoch (selected on the trained dataset) is starred. Read-only summary —
        colon/endo rows are cross-task transfer numbers, not their own task scores."""
        if not self._monitor_history:
            return

        def _f(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and not math.isnan(x) else "  n/a"

        prim_ds = self._val_manager._dataset_name
        series  = [(prim_ds, self._epoch_history)]
        series += [(_d, _h) for _d, _h in self._monitor_history.items()]

        _rank0_print(f"\n{'='*104}")
        _rank0_print("CROSS-DATASET VALIDATION SUMMARY  (all metrics · all epochs · original mode)")
        _rank0_print(
            f"best epoch (selected on {prim_ds} / {self._primary_metric}): "
            f"{self._best_epoch}  (score={self._best_score:.4f})"
        )
        _rank0_print(
            "colon/endo rows below are CROSS-TASK transfer (chest-trained model), "
            "not own-task scores."
        )
        _rank0_print(f"{'='*104}")
        _rank0_print(
            f"{'Dataset':<10} {'Epoch':<6} {'macro_f1':<10} "
            f"{'acc_aACC':<10} {'mAP':<10} {'AUC':<10} {'Best?':<6}"
        )
        _rank0_print(f"{'-'*104}")
        for ds, hist in series:
            for e in hist:
                am   = e.get("auc_map", {}) or {}
                o    = e.get("modes", {}).get("original", {})
                ep   = e.get("epoch")
                auc  = am.get("macro_auc", am.get("AUC", float("nan")))
                star = "★" if (ds == prim_ds and ep == self._best_epoch) else ""
                _rank0_print(
                    f"{ds:<10} {str(ep):<6} "
                    f"{_f(o.get('macro_f1', float('nan'))):<10} "
                    f"{_f(am.get('accuracy_aacc', float('nan'))):<10} "
                    f"{_f(am.get('mAP', float('nan'))):<10} "
                    f"{_f(auc):<10} {star:<6}"
                )
            _rank0_print(f"{'-'*104}")
        _rank0_print(f"{'='*104}\n")
