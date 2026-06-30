#!/usr/bin/env python3
"""Top-level inference pipeline entrypoint."""

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.environment import setup_environment
setup_environment()

from config.train_config.training_configs import ModelType, TrainingConfig
from models.symbolAdapter.symbol_manager import SymbolManager
from models.symbolAdapter.validation import ValidationManager
from models.symbolAdapter.dspo_module import setup_dspo_model, DspoModule
from models.symbolAdapter.symbol_router import SymbolRouter
from dataload.data_utils import create_combined_dataloader, load_datasets_for_config
from train import (
    extract_dataset_labels,
    extract_dataset_labels_dict,
    setup_tokenizer_and_processor,
)


class InferenceOrchestrator:
    """Orchestrates comprehensive inference evaluation using ValidationManager."""

    def __init__(
        self,
        checkpoint_path: str,
        dataset_type: str,
        model_type: str,
        device: str = "cuda:0",
        max_val_samples: int = 0,
        num_examples: int = 5,
        run_name: str = "",
        output_dir: Optional[str] = None,
        validation_modes: Optional[str] = None,
        num_workers: int = 2,
        symbol_map_file: Optional[str] = None,
        metrics_dir: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.dataset_type = dataset_type
        self.model_type = model_type.lower()
        self.device = device
        self.max_val_samples = max_val_samples
        self.num_examples = num_examples
        self.validation_modes = validation_modes
        self.run_name = run_name
        self.num_workers = num_workers
        self.symbol_map_file = symbol_map_file

        self.config = TrainingConfig()

        if output_dir:
            self.config.output_dir = output_dir
        if metrics_dir:
            self.config.metrics_dir = metrics_dir
        if run_name:
            self.config.run_name = run_name

        self.config.data_config.num_workers = num_workers

        self._setup_logging()

        self.model = None
        self.processor = None
        self.symbol_manager = None
        self.validator = None
        self.val_dataloader = None
        self._slot_symbol_map = None

    def _setup_logging(self):
        logs_dir = self.config.get_logs_dir()
        os.makedirs(logs_dir, exist_ok=True)

        log_file = os.path.join(logs_dir, f"{self.run_name}.log")

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
            ],
        )

        logging.info("Logging setup complete: %s", log_file)

    def load_checkpoint_and_config(self):
        try:
            checkpoint: Dict[str, Any] = {}

            if self.checkpoint_path:
                logging.info("Loading checkpoint: %s", self.checkpoint_path)

                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )

                if "config" not in checkpoint:
                    raise ValueError("Checkpoint missing configuration")

                old_config = checkpoint["config"]

                output_dir = self.config.output_dir
                run_name = self.config.run_name

                self.config = old_config
                self.config.output_dir = output_dir
                self.config.run_name = run_name

            else:
                self.config.model_type = ModelType(self.model_type)

            self.config.data_config.dataset_type = self.dataset_type
            self.config.data_config.max_samples = self.max_val_samples

            # validation_modes is the only legitimate override at inference time;
            # all other symbol settings come from the saved checkpoint config
            if self.validation_modes:
                self.config.symbol_config.validation_modes = self.validation_modes

            # Auto-detect D-SPO from checkpoint — no flag needed
            if checkpoint.get("router_state") is not None:
                self.config.diff_symbol_config.enabled = True
                logging.info("D-SPO detected in checkpoint — will restore router")

            return checkpoint

        except Exception as exc:
            logging.error("Failed to load checkpoint: %s", exc)
            raise

    def setup_model_and_data(self, checkpoint):
        try:
            tokenizer, processor = setup_tokenizer_and_processor(self.config)
            self.processor = processor  # ✅ store processor for ValidationManager

            dataset_labels = extract_dataset_labels(self.config)
            labels_per_dataset = extract_dataset_labels_dict(self.config)

            self.symbol_manager = SymbolManager(
                original_labels=dataset_labels,
                tokenizer=tokenizer,
                dynamic_per_epoch=False,
                token_size=self.config.symbol_config.symbol_token_size,
                no_symbols=self.config.symbol_config.no_symbols,
                swap_labels=self.config.symbol_config.swap_labels,
                labels_per_dataset=labels_per_dataset,
            )

            # Load and normalize symbol mappings to per-dataset format {ds_name: {label: symbol}}
            self.current_mappings: Dict[str, Dict[str, str]] = {}
            if not (self.config.symbol_config.no_symbols and not self.config.symbol_config.swap_labels):
                from config.data_config.master_config import DATASET_CONFIGS as _DC_init
                raw = checkpoint.get("symbol_mappings", {}).get("current_epoch_mappings", {})
                if raw:
                    first_val = next(iter(raw.values()), None)
                    if isinstance(first_val, dict):
                        self.current_mappings = raw  # already per-dataset (new format)
                    else:
                        # old flat format → convert to per-dataset
                        for ds_type, ds_cfg in _DC_init.items():
                            ds_name = ds_type.value
                            labels = list(ds_cfg.valid_labels or [])
                            self.current_mappings[ds_name] = {l: raw[l] for l in labels if l in raw}

            self.config.data_config.split = "test"
            self.config.data_config.max_samples = self.max_val_samples

            _, test_datasets = load_datasets_for_config(
                self.config,
                inference_mode=True,
            )

            if not test_datasets:
                raise ValueError(
                    f"No test datasets were loaded for {self.dataset_type}"
                )

            self.val_dataloader = create_combined_dataloader(
                test_datasets,
                processor,
                self.config,
                shuffle=False,
                num_examples=self.num_examples,
            )

            apply_lora = bool(self.checkpoint_path)

            if self.model_type == "qwen":
                from models.backends.custom_qwen import CustomQwen

                self.model = CustomQwen(
                    model_path=getattr(
                        self.config,
                        "qwen_model_name",
                        "Qwen/Qwen2-Audio-7B-Instruct",
                    ),
                    device=self.device,
                    lora=apply_lora,
                    lora_rank=self.config.lora_config.rank,
                    lora_alpha=self.config.lora_config.alpha,
                    lora_dropout=self.config.lora_config.dropout,
                )

            elif self.model_type == "salmonn":
                from models.backends.custom_salmonn import CustomSalmonn

                self.model = CustomSalmonn(
                    device=self.device,
                    lora=apply_lora,
                    lora_rank=self.config.lora_config.rank,
                    lora_alpha=self.config.lora_config.alpha,
                    lora_dropout=self.config.lora_config.dropout,
                    low_resource=False,
                )

            elif self.model_type == "flamingo":
                from models.backends.custom_flamingo import CustomFlamingo

                self.model = CustomFlamingo(
                    model_path=getattr(
                        self.config,
                        "flamingo_model_name",
                        "nvidia/audio-flamingo-3-hf",
                    ),
                    device=self.device,
                    lora=apply_lora,
                    lora_rank=self.config.lora_config.rank,
                    lora_alpha=self.config.lora_config.alpha,
                    lora_dropout=self.config.lora_config.dropout,
                )

            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            # Resize embeddings for D-SPO slot tokens before loading weights
            if self.config.diff_symbol_config.enabled:
                setup_dspo_model(self.model, tokenizer, self.config)

            if "model_state" in checkpoint:
                self.model.load_state_dict(
                    checkpoint["model_state"],
                    strict=False,
                )

            self.model.to(self.device)
            self.model.eval()

            # Restore D-SPO router from checkpoint
            if self.config.diff_symbol_config.enabled:
                router_state = checkpoint.get("router_state")
                if router_state is not None:
                    cfg = self.config.diff_symbol_config
                    dummy_indices = [[0] * cfg.slot_vocab_size for _ in range(cfg.num_slots)]
                    router = SymbolRouter(
                        num_slots=cfg.num_slots,
                        slot_vocab_size=cfg.slot_vocab_size,
                        slot_vocab_indices=dummy_indices,
                        initial_tau=cfg.tau,
                        tau_min=cfg.tau_min,
                    )
                    router.load_state_dict(router_state)
                    router.eval()
                    slot_token_ids = torch.tensor(
                        [tokenizer.convert_tokens_to_ids(f"<slot_{i}>") for i in range(cfg.num_slots)],
                        dtype=torch.long,
                    )
                    dspo_module = DspoModule(slot_token_ids).to(self.device)
                    self.model.router = router
                    self.model.dspo_module = dspo_module
                    logging.info("D-SPO router restored from checkpoint (%d slots, K=%d)", cfg.num_slots, cfg.slot_vocab_size)

                    # Use saved Phase 2 label→symbol map if available; otherwise decode from router
                    from config.data_config.master_config import DATASET_CONFIGS, DatasetType as _DT
                    from models.symbolAdapter.symbol_manager import compute_slot_offsets
                    train_ds_names = set(self.config.data_config.dataset_type.split("-"))

                    phase2_map = checkpoint.get("dspo_phase2_label_map")
                    if phase2_map:
                        first_val = next(iter(phase2_map.values()))
                        if isinstance(first_val, dict):
                            self.current_mappings = phase2_map  # already per-dataset
                        else:
                            # old flat format → convert to per-dataset
                            for ds_name in train_ds_names:
                                try:
                                    labels = list(DATASET_CONFIGS[_DT(ds_name)].valid_labels)
                                except (ValueError, KeyError):
                                    continue
                                self.current_mappings[ds_name] = {l: phase2_map[l] for l in labels if l in phase2_map}
                        total = sum(len(m) for m in self.current_mappings.values())
                        logging.info("D-SPO: using saved phase2_label_map (%d labels across %d datasets)", total, len(self.current_mappings))
                    else:
                        # decode per-dataset label→symbol from router (Phase 0/1 path)
                        slot_offsets = compute_slot_offsets(train_ds_names)
                        for ds_name in train_ds_names:
                            try:
                                labels = list(DATASET_CONFIGS[_DT(ds_name)].valid_labels)
                            except (ValueError, KeyError):
                                continue
                            offset = slot_offsets.get(ds_name, 0)
                            slot_indices = list(range(offset, offset + len(labels)))
                            vocab_indices, _ = router.get_slot_mappings(slot_indices, hard=True, deterministic=True)
                            self.current_mappings[ds_name] = {
                                label: tokenizer.decode(idx.tolist()).strip() or f"<tok_{idx.tolist()}>"
                                for label, idx in zip(labels, vocab_indices)
                            }
                        total = sum(len(m) for m in self.current_mappings.values())
                        logging.info("D-SPO: decoded symbol_map from router (%d labels across %d datasets)", total, len(self.current_mappings))
                    # store slot pool for cross-task extension in run_comprehensive_inference
                    self._slot_symbol_map = checkpoint.get("dspo_slot_symbol_map")
                else:
                    logging.warning("D-SPO enabled but no router_state in checkpoint — router not restored")

            self.validator = ValidationManager(
                config=self.config,
                symbol_manager=self.symbol_manager,
                tokenizer=tokenizer,
                processor=processor,  # ✅ pass processor
            )

        except Exception as exc:
            logging.error("Failed to setup model and data: %s", exc)
            raise

    def run_comprehensive_inference(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, Any], List[Dict[str, Any]]]:

        self.config.inference_mode = True

        symbol_map: Dict[str, Dict[str, str]] = dict(self.current_mappings)
        if self._slot_symbol_map:
            from config.data_config.master_config import DATASET_CONFIGS as _DC
            pool = list(self._slot_symbol_map.values())
            train_ds_names = set(self.config.data_config.dataset_type.split("-"))
            for ds_type, ds_cfg in _DC.items():
                ds_name = ds_type.value
                labels = list(ds_cfg.valid_labels or [])
                if not labels:
                    continue
                if ds_name in train_ds_names:
                    symbol_map[ds_name] = dict(self.current_mappings.get(ds_name, {}))
                else:
                    chosen = random.sample(pool, k=min(len(labels), len(pool)))
                    symbol_map[ds_name] = dict(zip(labels, chosen))
            logging.info("Inference: extended symbol_map for cross-task from slot pool (%d slots)", len(pool))

        results = self.validator.run_comprehensive_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=0,
            symbol_map=symbol_map,
        )

        validation_scores = {}
        detailed_metrics = {}
        all_predictions = []

        for mode, ds_metrics in results["all_modes"].items():
            for ds_name, m in ds_metrics.items():

                key = f"{mode}_{ds_name}"

                validation_scores[key] = m["score"]
                detailed_metrics[key] = m["detailed"]

                if m.get("predictions"):
                    for p in m["predictions"]:
                        p["validation_mode"] = mode

                    all_predictions.extend(m["predictions"])

        return validation_scores, detailed_metrics, all_predictions

    def save_results(
        self,
        detailed_metrics: Dict[str, Any],
        all_predictions: List[Dict[str, Any]],
    ):
        logger = logging.getLogger(__name__)

        logger.info("=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)

        metrics_dir = self.config.get_metrics_dir()
        os.makedirs(metrics_dir, exist_ok=True)

        metrics_file = os.path.join(
            metrics_dir,
            f"{self.run_name}_metrics.json",
        )

        with open(metrics_file, "w") as handle:
            json.dump(
                detailed_metrics,
                handle,
                indent=2,
                default=str,
            )

        logger.info("Metrics saved: %s", metrics_file)

        predictions_file = os.path.join(
            metrics_dir,
            f"{self.run_name}_predictions.json",
        )

        with open(predictions_file, "w") as handle:
            json.dump(
                all_predictions,
                handle,
                indent=2,
                default=str,
            )

        logger.info("Predictions saved: %s", predictions_file)
        logger.info("=" * 80)

    def _apply_symbol_map_file(self):
        """Override current_mappings from a JSON file if --symbol_map_file was given."""
        if not self.symbol_map_file:
            return
        with open(self.symbol_map_file) as f:
            override = json.load(f)
        # Merge: file entries win, checkpoint entries fill the rest
        for ds_name, sym_dict in override.items():
            self.current_mappings[ds_name] = sym_dict
        logging.info("Symbol map overridden from %s: %s",
                     self.symbol_map_file,
                     {ds: list(v.keys()) for ds, v in override.items()})

    def run_complete_inference(self):

        checkpoint = self.load_checkpoint_and_config()
        self.setup_model_and_data(checkpoint)
        self._apply_symbol_map_file()

        validation_scores, detailed_metrics, all_predictions = (
            self.run_comprehensive_inference()
        )

        self.save_results(detailed_metrics, all_predictions)

        return {
            "validation_results": validation_scores,
            "detailed_metrics": detailed_metrics,
            "predictions_count": len(all_predictions),
            "run_name": self.run_name,
        }


def main():

    parser = argparse.ArgumentParser(description="Orchestrator Inference Pipeline")

    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["salmonn", "qwen", "flamingo"],
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--validation_modes", type=str, default=None,
                        help="Override validation modes from checkpoint (e.g. original,fixed,fresh)")
    parser.add_argument("--symbol_map_file", type=str, default=None,
                        help="JSON file with symbol overrides {ds_name: {label: symbol}}; merged over checkpoint mappings")
    parser.add_argument("--metrics_dir", type=str, default=None,
                        help="Directory for metrics/predictions output (dated subfolder recommended)")

    args = parser.parse_args()

    try:
        orchestrator = InferenceOrchestrator(
            checkpoint_path=args.checkpoint_path,
            dataset_type=args.dataset_type,
            model_type=args.model_type,
            device=args.device,
            max_val_samples=args.max_val_samples,
            num_examples=args.num_examples,
            run_name=args.run_name,
            output_dir=args.output_dir,
            validation_modes=args.validation_modes,
            num_workers=args.num_workers,
            symbol_map_file=args.symbol_map_file,
            metrics_dir=args.metrics_dir,
        )

        results = orchestrator.run_complete_inference()

        print("")
        print("=" * 80)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results saved as: {results['run_name']}")
        print(f"Predictions collected: {results['predictions_count']}")
        print("=" * 80)

        return 0

    except Exception as exc:
        print(f"Inference failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())