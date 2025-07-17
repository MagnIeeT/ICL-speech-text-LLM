#!/usr/bin/env python3
"""
Orchestrator Inference Pipeline
Evaluates trained models using comprehensive validation with detailed metrics and predictions
"""

import os
import sys
import json
import logging
import argparse
import torch
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

ICL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ICL_ROOT)

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "models"))

# Import required modules
from models.mlp_salmonn import MLPSalmonn
from models.symbolAdapter.configs.training_configs import TrainingConfig,SymbolMode
from models.symbolAdapter.symbol_manager import SymbolManager
from models.symbolAdapter.training.validation import ValidationManager
from models.symbolAdapter.orchestrator_training import load_datasets_for_config, create_combined_dataloader
from utils.evaluation_utils import evaluate_predictions
from data.master_config import DatasetType

class InferenceOrchestrator:
    """Orchestrates comprehensive inference evaluation using ValidationManager"""
    
    def __init__(
        self,
        checkpoint_path: str,
        dataset_type: str,
        device: str = "cuda:0",
        num_examples: int = 5,
        max_val_samples: int = 0,  # 0 = all samples
        output_dir: Optional[str] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.dataset_type = dataset_type
        self.device = device
        self.max_val_samples = max_val_samples
        self.num_examples = num_examples

        # Setup output directories
        self.results_base = output_dir or "/data1/chandnia/neeraja/results/model_ICL"
        self.metrics_dir = os.path.join(self.results_base, "orchestrator_metrics")
        self.logs_dir = os.path.join(self.results_base, "orchestrator_logs")
        
        # Create directories
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.metrics_output_dir = os.path.join(self.metrics_dir, current_date)
        self.logs_output_dir = os.path.join(self.logs_dir, current_date)
        os.makedirs(self.metrics_output_dir, exist_ok=True)
        os.makedirs(self.logs_output_dir, exist_ok=True)
        
        # Generate run name
        timestamp = datetime.now().strftime("%d%m_%H%M")
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '').replace('.pth', '')
        self.run_name = f"{timestamp}_inference_{checkpoint_name}_{dataset_type}"
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.config = None
        self.symbol_manager = None
        self.validator = None
        self.val_dataloader = None
        
        logging.info(f"🚀 Initializing Inference Orchestrator")
        logging.info(f"📁 Checkpoint: {checkpoint_path}")
        logging.info(f"📊 Dataset: {dataset_type}")
        logging.info(f"🔧 Device: {device}")
        logging.info(f"📝 Run Name: {self.run_name}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.logs_output_dir, f"{self.run_name}.log")
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Setup logging format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                # logging.FileHandler(log_file),  
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"📋 Logging setup complete: {log_file}")
    
    def load_checkpoint_and_config(self):
        """Load trained model checkpoint and configuration"""
        logging.info("=" * 80)
        logging.info("LOADING CHECKPOINT AND CONFIGURATION")
        logging.info("=" * 80)
        
        try:
            # Load checkpoint
            logging.info(f"Loading checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Extract configuration
            if 'config' in checkpoint:
                self.config = checkpoint['config']
                logging.info("✅ Configuration loaded from checkpoint")
            else:
                logging.error("❌ No configuration found in checkpoint")
                raise ValueError("Checkpoint missing configuration")
            
            # Log checkpoint info
            if 'step_info' in checkpoint:
                step_info = checkpoint['step_info']
                logging.info(f"📋 Checkpoint Info:")
                logging.info(f"  Phase: {step_info.get('phase', 'unknown')}")
                logging.info(f"  Step: {step_info.get('step_id', 'unknown')}")
                logging.info(f"  Cycle: {step_info.get('cycle', 'unknown')}")
                logging.info(f"  Epoch: {step_info.get('epoch', 'unknown')}")
                logging.info(f"  Description: {step_info.get('description', 'N/A')}")
            
            # Update config for inference
            self.config.data_config.dataset_type = self.dataset_type

            self.config.data_config.max_samples = self.max_val_samples
            
            return checkpoint
            
        except Exception as e:
            logging.error(f"❌ Failed to load checkpoint: {str(e)}")
            raise
    
    def setup_model_and_data(self, checkpoint):
        """Setup model and data loaders using same logic as training orchestrator"""
        logging.info("=" * 80)
        logging.info("SETTING UP MODEL AND DATA")
        logging.info("=" * 80)
        
        try:
            # Setup tokenizer and other components
            from models.symbolAdapter.orchestrator_training import (
                setup_tokenizer,
                extract_dataset_labels,
                extract_dataset_labels_dict
            )
            
            # Setup tokenizer (same as training)
            tokenizer = setup_tokenizer()
            logging.info("✅ Tokenizer setup complete")
            
            # Extract dataset labels (same as training)
            dataset_labels = extract_dataset_labels(self.config)
            logging.info(f"dataset_labels : {dataset_labels}")
            dataset_labels_dict = extract_dataset_labels_dict(self.config)
            logging.info(f"dataset_labels : {dataset_labels_dict}")

            # Setup symbol manager (same as training)
            if 'symbol_mappings' in checkpoint:
                symbol_data = checkpoint['symbol_mappings']
                logging.info("📋 Restoring symbol mappings from checkpoint...")
                current_mappings = symbol_data['current_epoch_mappings']
                logging.info(f"✅ Restored symbol mappings: {current_mappings}")
            else:
                logging.warning("⚠️ No symbol mappings found in checkpoint, using default setup")
            
            self.symbol_manager = SymbolManager(
                    original_labels=dataset_labels,
                    original_labels_dict=dataset_labels_dict,
                    tokenizer=tokenizer,
                    dynamic_per_epoch=False,
                    symbol_type=self.config.symbol_config.symbol_type
                )

            self.current_mappings = current_mappings

            # logging.info(f"✅ SymbolManager initialized with {len(self.symbol_manager.symbols)} symbols")

            logging.info(f"current_mappings: {self.current_mappings}")            

            # ✅ FIX: Update config for TEST split inference
            self.config.data_config.split = 'test'  # Force test split for inference

            self.config.data_config.max_samples = self.max_val_samples
            
            self.config.data_config.num_examples = self.num_examples
            
            # Load datasets (same as training orchestrator)
            logging.info(f"📊 Loading datasets for: {self.dataset_type} (TEST split)")
            train_datasets, test_datasets = load_datasets_for_config(self.config, inference_mode=True)
            
            
            # ✅ FIX: Create processor with correct model type
            from data.model_processors import get_processor
            processor = get_processor(self.config.model_type.value, tokenizer=tokenizer)  # ✅ Use model_type.value
            logging.info("✅ Processor initialized")
            
            # ✅ FIX: Create test dataloader (not validation)
            self.val_dataloader = create_combined_dataloader(
                test_datasets, 
                processor,
                self.config, 
                num_examples=self.num_examples,
                shuffle=False
            )
            
            logging.info(f"✅ Test dataloader created: {len(self.val_dataloader)} batches")
            
            # Get initial symbol mappings
            initial_symbol_mappings = self.symbol_manager.get_symbols_for_epoch(0)
            
            # ✅ FIX: Initialize model with correct config attributes
            logging.info("🤖 Initializing model...")
            self.model = MLPSalmonn(
                device="cuda:0",
                # label_tokens=list(initial_symbol_mappings.values()),
                # hidden_dim=self.config.mlp_config.hidden_dim,  # ✅ mlp_config.hidden_dim
                # dropout=self.config.mlp_config.dropout,        # ✅ mlp_config.dropout
                lora=True,
                lora_rank=self.config.lora_config.rank,        # ✅ lora_config.rank
                lora_alpha=self.config.lora_config.alpha,      # ✅ lora_config.alpha
                lora_dropout=self.config.lora_config.dropout,  # ✅ lora_config.dropout
                low_resource=False,
                # use_output_mlp=self.config.mlp_config.use_output_mlp,  # ✅ mlp_config.use_output_mlp
                # bypass_mlp=not self.config.mlp_config.use_input_mlp    # ✅ NOT use_input_mlp = bypass_mlp
            )
            
            # Load model state from checkpoint
            logging.info("📥 Loading model state from checkpoint...")
            if 'model_state' in checkpoint:
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        checkpoint['model_state'], 
                        strict=False
                    )
                    
                    loaded_params = len(checkpoint['model_state']) - len(unexpected_keys)
                    logging.info(f"✅ Loaded {loaded_params} model parameters")
                    
                    if missing_keys:
                        logging.info(f"⚠️ Missing keys: {len(missing_keys)} (expected for frozen params)")
                    if unexpected_keys:
                        logging.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                        
                except Exception as e:
                    logging.error(f"❌ Failed to load model state: {str(e)}")
                    raise
            else:
                logging.warning("⚠️ No model state found in checkpoint")
        
        # Move model to device and set eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # ✅ FIX: Setup ValidationManager with correct config attribute
            self.validator = ValidationManager(
                config=self.config,
                symbol_manager=self.symbol_manager,
                tokenizer=tokenizer
            )
            
            logging.info("✅ ValidationManager initialized")
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error(f"❌ Failed to setup model and data: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def run_comprehensive_inference(self) -> Tuple[Dict[str, float], Dict[str, Any], List[Dict[str, Any]]]:
        """Run comprehensive inference using ValidationManager"""
        
        # Enable inference mode
        self.config.inference_mode = True
        
        # Run validation
        results = self.validator.run_comprehensive_validation(
            config=self.config,
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=0,
            phase='lora',  # Use 'joint' for comprehensive inference
            symbol_mappings = self.current_mappings,
        )
        
        # Extract and return (ValidationManager did all the work)
        return results['validation_scores'], results['detailed_metrics'], results['all_predictions']


    def save_results(self, detailed_metrics: Dict[str, Any], all_predictions: List[Dict[str, Any]]):
        """Save detailed metrics and predictions to files"""
        logging.info("=" * 80)
        logging.info("SAVING RESULTS")
        logging.info("=" * 80)
        
        try:
            # Save metrics.json
            metrics_file = os.path.join(self.metrics_output_dir, f"{self.run_name}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(detailed_metrics, f, indent=2, default=str)
            
            logging.info(f"✅ Metrics saved: {metrics_file}")
            
            # Save predictions.json
            predictions_file = os.path.join(self.metrics_output_dir, f"{self.run_name}_predictions.json")
            with open(predictions_file, 'w') as f:
                json.dump(all_predictions, f, indent=2, default=str)
            
            logging.info(f"✅ Predictions saved: {predictions_file}")
            
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error(f"❌ Failed to save results: {str(e)}")
            raise
    
    
    def run_complete_inference(self):
        """Run complete inference pipeline"""
        try:
            logging.info("🚀 Starting Orchestrator Inference Pipeline")
            
            # 1. Load checkpoint and configuration
            checkpoint = self.load_checkpoint_and_config()
            
            # 2. Setup model and data
            self.setup_model_and_data(checkpoint)
            
            # 3. Run comprehensive inference
            validation_scores, detailed_metrics, all_predictions = self.run_comprehensive_inference()
            
            # 4. Save results
            self.save_results(detailed_metrics, all_predictions)
            
            # 5. Simple final logging (inline)
            logging.info("=" * 60)
            logging.info("FINAL INFERENCE RESULTS")
            logging.info("=" * 60)
            for mode, score in validation_scores.items():
                if not mode.endswith('_loss'):
                    if isinstance(score, str) and '|' in score and ':' in score:
                        logging.info(f"{mode:<20}:")
                        parts = score.split('|')
                        for p in parts:
                            key, val = p.split(':')
                            logging.info(f"{'':<20}  {key.strip():<15}: {float(val):.4f}")
                    else:
                        logging.info(f"{mode:<20}: {float(score):.4f}")
            logging.info("=" * 60)
            
            logging.info("✅ Orchestrator Inference Pipeline completed successfully!")
            
            return {
                'validation_results': validation_scores,
                'detailed_metrics': detailed_metrics,
                'predictions_count': len(all_predictions),
                'run_name': self.run_name
            }
            
        except Exception as e:
            logging.error(f"❌ Inference pipeline failed: {str(e)}")
            raise

def main():
    """Main function for orchestrator inference"""
    parser = argparse.ArgumentParser(description="Orchestrator Inference Pipeline")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--dataset_type", type=str, required=True,
                       help="Dataset type for evaluation (e.g., voxceleb, hvb)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for inference")
    parser.add_argument("--max_val_samples", type=int, default=0,
                       help="Maximum validation samples (0 = all)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--num_examples", type=int, default=5)

    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        return 1
    
    try:
        # Initialize orchestrator
        orchestrator = InferenceOrchestrator(
            checkpoint_path=args.checkpoint_path,
            dataset_type=args.dataset_type,
            device=args.device,
            num_examples=args.num_examples,
            max_val_samples=args.max_val_samples,
            output_dir=args.output_dir
        )
        
        # Run inference
        results = orchestrator.run_complete_inference()
        
        print(f"✅ Inference completed successfully!")
        print(f"📁 Results saved as: {results['run_name']}")
        print(f"📊 Predictions collected: {results['predictions_count']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())