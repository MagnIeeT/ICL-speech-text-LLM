#!/usr/bin/env python3
"""Top-level inference pipeline entrypoint."""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from config.train_config.training_configs import ModelType, TrainingConfig
from models.symbolAdapter.symbol_manager import SymbolManager
from models.symbolAdapter.validation import ValidationManager
from train import (
    create_combined_dataloader,
    extract_dataset_labels,
    extract_dataset_labels_dict,
    load_datasets_for_config,
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
        no_symbols: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.dataset_type = dataset_type
        self.model_type = model_type.lower()
        self.device = device
        self.max_val_samples = max_val_samples
        self.num_examples = num_examples
        self.no_symbols = no_symbols

        self.results_base = output_dir or os.path.join(os.getcwd(), "results")
        self.metrics_dir = os.path.join(self.results_base, "orchestrator_metrics")
        self.logs_dir = os.path.join(self.results_base, "orchestrator_logs")

        current_date = datetime.now().strftime("%Y-%m-%d")
        self.metrics_output_dir = os.path.join(self.metrics_dir, current_date)
        self.logs_output_dir = os.path.join(self.logs_dir, current_date)
        os.makedirs(self.metrics_output_dir, exist_ok=True)
        os.makedirs(self.logs_output_dir, exist_ok=True)

        self.run_name = run_name
        self._setup_logging()

        self.model = None
        self.config = None
        self.symbol_manager = None
        self.validator = None
        self.val_dataloader = None

    def _setup_logging(self):
        log_file = os.path.join(self.logs_output_dir, f"{self.run_name}.log")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Logging setup complete: %s", log_file)

    def load_checkpoint_and_config(self):
        try:
            checkpoint: Dict[str, Any] = {}
            if self.checkpoint_path:
                logging.info("Loading checkpoint: %s", self.checkpoint_path)
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

                if "config" not in checkpoint:
                    raise ValueError("Checkpoint missing configuration")
                self.config = checkpoint["config"]
            else:
                self.config = TrainingConfig()
                self.config.model_type = ModelType(self.model_type)

            self.config.data_config.dataset_type = self.dataset_type
            self.config.data_config.max_samples = self.max_val_samples
            self.config.symbol_config.no_symbols = self.no_symbols
            if self.no_symbols:
                self.config.symbol_config.dynamic_symbols = False

            return checkpoint

        except Exception as exc:
            logging.error("Failed to load checkpoint: %s", exc)
            raise

    def setup_model_and_data(self, checkpoint):
        try:
            tokenizer, processor = setup_tokenizer_and_processor(self.config)
            dataset_labels = extract_dataset_labels(self.config)
            _ = extract_dataset_labels_dict(self.config)

            if "symbol_mappings" in checkpoint:
                current_mappings = checkpoint["symbol_mappings"].get("current_epoch_mappings", {})
            else:
                current_mappings = {}
            self.symbol_manager = SymbolManager(
                original_labels=dataset_labels,
                tokenizer=tokenizer,
                dynamic_per_epoch=False,
                symbol_type=self.config.symbol_config.symbol_type,
                no_symbols=self.config.symbol_config.no_symbols,
            )
            self.current_mappings = {} if self.config.symbol_config.no_symbols else current_mappings

            self.config.data_config.split = "test"
            self.config.data_config.max_samples = self.max_val_samples

            _, test_datasets = load_datasets_for_config(self.config, inference_mode=True)
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
                    model_path=getattr(self.config, "qwen_model_name", "Qwen/Qwen2-Audio-7B-Instruct"),
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
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            if "model_state" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state"], strict=False)

            self.model.to(self.device)
            self.model.eval()

            self.validator = ValidationManager(
                config=self.config,
                symbol_manager=self.symbol_manager,
                tokenizer=tokenizer,
            )

        except Exception as exc:
            logging.error("Failed to setup model and data: %s", exc)
            raise

    def run_comprehensive_inference(self) -> Tuple[Dict[str, float], Dict[str, Any], List[Dict[str, Any]]]:
        self.config.inference_mode = True
        results = self.validator.run_comprehensive_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=0,
            phase="lora",
            symbol_mappings=self.current_mappings,
        )
        return results["validation_scores"], results["detailed_metrics"], results["all_predictions"]

    def save_results(self, detailed_metrics: Dict[str, Any], all_predictions: List[Dict[str, Any]]):
        metrics_file = os.path.join(self.metrics_output_dir, f"{self.run_name}_metrics.json")
        with open(metrics_file, "w") as handle:
            json.dump(detailed_metrics, handle, indent=2, default=str)

        predictions_file = os.path.join(self.metrics_output_dir, f"{self.run_name}_predictions.json")
        with open(predictions_file, "w") as handle:
            json.dump(all_predictions, handle, indent=2, default=str)

    def run_complete_inference(self):
        checkpoint = self.load_checkpoint_and_config()
        self.setup_model_and_data(checkpoint)
        validation_scores, detailed_metrics, all_predictions = self.run_comprehensive_inference()
        self.save_results(detailed_metrics, all_predictions)

        return {
            "validation_results": validation_scores,
            "detailed_metrics": detailed_metrics,
            "predictions_count": len(all_predictions),
            "run_name": self.run_name,
        }


def main():
    parser = argparse.ArgumentParser(description="Orchestrator Inference Pipeline")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to trained model checkpoint")
    parser.add_argument("--dataset_type", type=str, required=True, help="Dataset type for evaluation")
    parser.add_argument("--model_type", type=str, required=True, choices=["salmonn", "qwen"], help="Model type")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for inference")
    parser.add_argument("--max_val_samples", type=int, default=0, help="Maximum validation samples (0 = all)")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--no_symbols", action="store_true")

    args = parser.parse_args()

    if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found: {args.checkpoint_path}")
        return 1

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
            no_symbols=args.no_symbols,
        )
        results = orchestrator.run_complete_inference()
        print("Inference completed successfully")
        print(f"Results saved as: {results['run_name']}")
        print(f"Predictions collected: {results['predictions_count']}")
        return 0
    except Exception as exc:
        print(f"Inference failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



# 1. See you are passing symbol manager in initialise model, get_symbolfor epcoh is not there in symbol_mamanger and that should we called when no symbol is true
# 1. swap otpion in symbol manager, it basically mean jumble the labme positve means negative and negative means postive, this can we possible for fixed symbols and no symbole
# 2. validation.py optimise.
# 3. remove unnnecessaary function/metho from symbol manager
# 4. If during validation i just want to do inference on original symbols or both or new symbols. should I add that in config and update validatin.py accordingly
# 5. you can move load dataset for config and dataloader functin in data_utils
# need to remove hardcoded path while pushing code to repo. (put placeholder)