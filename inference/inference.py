import os
import sys
import logging
import argparse
import json
import time
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import ModelFactory
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType
from utils.training_utils import setup_logging, load_checkpoint
from utils.evaluation_utils import evaluate_predictions, save_evaluation_results, clean_prediction
from utils.performance_utils import log_gpu_memory_usage, log_system_info, time_function, PerformanceTracker
from utils.data_utils import load_dataset
from data.model_processors import get_processor
from config.inference_config import get_inference_config

logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments for inference.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run inference with ICL models")
    
    # Required arguments from inference.sh
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--today", type=str, default=datetime.datetime.now().strftime("%Y-%m-%d"), 
                        help="Date for output directory")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output files")
    
    # Dataset arguments
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset to use")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to use for inference")
    parser.add_argument("--input_mode", type=str, default="speech_only", 
                        choices=["speech_only", "text_only", "speech_and_text"],
                        help="Input mode for the model")
    parser.add_argument("--fewshot_mode", type=str, default="text", 
                        choices=["text", "speech"],
                        help="Mode for few-shot examples")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="salmonn", help="Type of model to use")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--save_per_dataset", action="store_true", help="Save separate result files for each dataset type")
    
    # Performance options
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for inference")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for inference")
    parser.add_argument("--optimize_batch_size", action="store_true", help="Automatically find optimal batch size")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Maximum batch size to try when optimizing")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster inference")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Use pinned memory for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    
    # Debugging arguments
    parser.add_argument("--debug_samples", type=int, default=0, 
                        help="Number of samples to use for debugging (0 = use all samples)")
    
    # New argument for randomize swap
    parser.add_argument("--randomize_swap", type=bool, default=False,
                        help="Randomize swap configurations during inference")
    
    # Add new arguments
    parser.add_argument("--balance_datasets", type=bool, default=False,
                        help="Balance datasets during inference")
    parser.add_argument("--interleave", type=bool, default=False,
                        help="Interleave datasets during inference")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

@time_function
def run_inference(args):
    """
    Run inference with the specified model and dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dict: Results of inference
    """
    try:
        # Add detailed logging for dataset_type
        logger.info("=== Dataset Type Debug Info ===")
        logger.info(f"Raw dataset_type argument: '{args.dataset_type}'")
        # Change comma splitting to hyphen splitting
        if "-" in args.dataset_type:
            dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]
            logger.info(f"Running multi-task inference with datasets: {dataset_types}")
        else:
            dataset_types = [DatasetType(args.dataset_type)]
            logger.info(f"Running inference with dataset: {dataset_types}")

        # Set random seed
        # set_seed(args.seed)
        
        # Create output directory
        results_dir = f"/data2/neeraja/neeraja/results/model_ICL/metrics/{args.today}"
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Starting inference with arguments: {args}")
        
        # Log system information
        log_system_info()
        log_gpu_memory_usage("Initial GPU memory usage")
        
        # Initialize performance tracker
        performance_tracker = PerformanceTracker(log_interval=10)
        
        config = get_inference_config(args.model_type)
        # Create model
        logger.info(f"Creating model of type {args.model_type}")
        model = ModelFactory.create_model(
            model_type=args.model_type,
            multi_task=False,
            device=args.device,
            low_resource=True,
            **config.get("model_args", {})
        )
        
        # Only load checkpoint if peft_model_path is provided and not empty
        if args.peft_model_path and args.peft_model_path.strip():
            logger.info(f"Loading checkpoint from {args.peft_model_path}")
            checkpoint = load_checkpoint(args.peft_model_path, map_location=args.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                finetuned_state_dict = checkpoint["model_state_dict"]
                finetuned_keys = set(finetuned_state_dict.keys())
                logging.info(f"Updating {len(finetuned_keys)} parameters from finetuned model")
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
                finetuned_state_dict = checkpoint["state_dict"]
                finetuned_keys = set(finetuned_state_dict.keys())
                logging.info(f"Updating {len(finetuned_keys)} parameters from finetuned model")
            elif "model" in checkpoint:
                model.salmonn.load_state_dict(checkpoint["model"], strict=False)
                finetuned_state_dict = checkpoint['model'] 
                finetuned_keys = set(finetuned_state_dict.keys())
                logging.info(f"Updating {len(finetuned_keys)} parameters from finetuned model salmonn")
            else:
                model.load_state_dict(checkpoint, strict=False)
                logger.info("Loaded checkpoint directly as state dict")
        else:
            logger.info("No checkpoint path provided, using base model without loading weights")

        
        
        # Apply torch.compile if requested and available
        if args.compile and hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            try:
                logger.info("Applying torch.compile for faster inference")
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")
                logger.debug(traceback.format_exc())
        
        # Move model to device
        model.to(args.device)
        logger.info(f"Model loaded and moved to {args.device}")
        
        # Create model-specific processor
        
        logger.info(f"Creating {args.model_type} processor")
        if args.model_type == "salmonn":    
            model_processor = get_processor(args.model_type, model.input_processor,model.llama_tokenizer)
        else:
            model_processor = get_processor(args.model_type, model.input_processor)
        
        # Load datasets
        logger.info("Loading datasets")
        datasets = {}
        for dt in dataset_types:
            try:
                full_dataset = load_dataset(dt, split=args.split)
                if args.debug_samples and args.debug_samples > 0:
                    # Take only the specified number of samples for debugging
                    logger.info(f"DEBUG MODE: Limiting to {args.debug_samples} samples for dataset {dt}")
                    datasets[dt] = full_dataset.select(range(args.debug_samples))
                else:
                    datasets[dt] = full_dataset
                logger.info(f"Loaded dataset {dt}: {len(datasets[dt])} examples")
            except Exception as e:
                logger.error(f"Error loading dataset {dt}: {e}")
                logger.debug(traceback.format_exc())
                return 1
        
        # Create dataset and dataloader
        logger.info("Creating dataset and dataloader")
        dataset = DatasetFactory.create_dataset(
            dataset_type=dataset_types,
            dataset=datasets,
            processor=model_processor,
            is_training=False,
            input_mode=args.input_mode,
            fewshot_mode=args.fewshot_mode,
            num_examples=args.num_examples,
            random_examples=False,
            model_type=args.model_type,
            run_name=args.run_name,
            randomize_swap=args.randomize_swap,
            balance_datasets=args.balance_datasets,
            interleave=args.interleave
        )

        # Add debug logs here
        logger.info(f"=== Dataset Debug Info ===")
        logger.info(f"Dataset types requested: {dataset_types}")
        logger.info(f"Total dataset length: {len(dataset)}")
        if hasattr(dataset, 'datasets'):
            logger.info("Individual dataset lengths:")
            for dt, ds in dataset.datasets.items():
                logger.info(f"  - {dt}: {len(ds)} samples")
        logger.info("========================")
        
        # Optimize batch size if requested
        batch_size = args.batch_size
        if args.optimize_batch_size:
            logger.info("Finding optimal batch size for inference")
            # This would need to be implemented with a batch size optimizer
            # For now, just use the provided batch size
            logger.warning("Batch size optimization not implemented, using provided batch size")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=model_processor.collate_batch
        )
        
        # Limit samples if specified
        total_samples = len(dataset)
        if args.max_samples is not None and args.max_samples < total_samples:
            total_samples = args.max_samples
            logger.info(f"Limiting inference to {total_samples} samples")
        else:
            logger.info(f"Running inference on {total_samples} samples")
        
        # Run inference
        logger.info("Starting inference")
        results = []
        
        # Import gc at the top of the file
        import gc
        
        # Set up mixed precision if requested
        if args.bf16 and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            logger.info("Using bfloat16 for inference")
        elif args.fp16:
            amp_dtype = torch.float16
            logger.info("Using float16 for inference")
        else:
            amp_dtype = None
            logger.info("Using full precision for inference")
            
        autocast = torch.cuda.amp.autocast if amp_dtype else lambda: DummyContextManager()
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Running inference", position=0, leave=True, dynamic_ncols=True)
            for batch_idx, batch in enumerate(pbar):
                if args.max_samples is not None and batch_idx * batch_size >= args.max_samples:
                    break
                
                try:
                    # Clear memory every 50 iterations
                    if batch_idx > 0 and batch_idx % 50 == 0:
                        logger.info(f"Clearing memory at iteration {batch_idx}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        log_gpu_memory_usage("GPU memory after clearing")
                    
                    # Update progress bar description with current batch
                    pbar.set_description(f"Processing batch {batch_idx+1}")
                    
                    # Move batch to device
                    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # Print full prompt for first 5 iterations
                    if batch_idx < 5:
                        logger.info(f"=== Batch {batch_idx+1} Prompt ===")
                        if "prompt" in batch:
                            logger.info(batch["prompt"][0])
                        elif "input_ids" in batch and hasattr(model, "processor") and hasattr(model.processor, "tokenizer"):
                            try:
                                decoded_prompt = model.processor.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
                                logger.info(decoded_prompt)
                            except Exception as e:
                                logger.warning(f"Could not decode input_ids: {e}")
                        else:
                            logger.info("Prompt not available in batch")
                        logger.info("=" * 50)
                    
                    # Run inference
                    start_time = time.time()
                    # with autocast(dtype=amp_dtype) if amp_dtype else autocast():
                    outputs = model.generate_output(batch)
                    inference_time = time.time() - start_time
                    
                    # Process outputs and log them
                    for i, (output, true_label) in enumerate(zip(outputs, batch["completion"])):
                        cleaned_output = clean_prediction(output, batch["dataset_type"][i] if isinstance(batch["dataset_type"], list) else batch["dataset_type"])
                        logger.info(f"Batch {batch_idx+1}, Sample {i+1}:")
                        logger.info(f"Predicted (original): {output}")
                        logger.info(f"Predicted (cleaned): {cleaned_output}")
                        logger.info(f"True: {true_label}")
                        logger.info("-" * 50)


                        prediction = {
                            "text": batch["text"][i] if isinstance(batch["text"], list) else batch["text"],
                            "true_label": true_label,
                            "predicted_label (cleaned)": cleaned_output,
                            "predicted_label": output.strip(),
                            "dataset_type": batch["dataset_type"][i].value if isinstance(batch["dataset_type"], list) else batch["dataset_type"].value
                        }
                        
                        # Add question field for SQA dataset
                        if (isinstance(batch["dataset_type"], list) and batch["dataset_type"][i].value == "sqa") or \
                           (not isinstance(batch["dataset_type"], list) and batch["dataset_type"].value == "sqa"):
                            prediction["question"] = batch["question"][i] if isinstance(batch["question"], list) else batch["question"]



                        results.append(prediction)
                    
                    # Update performance tracker
                    performance_tracker.update(inference_time, len(batch["input_ids"]))
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    continue
        
        # Save final results
        logger.info(f"Inference complete. Processed {len(results)} samples")
        save_final_results(results, args, results_dir)
        
        # Log performance summary
        performance_tracker.log_summary()
        
        return {
            "results": results,
            "performance": performance_tracker.get_summary()
        }
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Inference failed: {str(e)}") from e



def save_final_results(results, args, results_dir):
    """
    Save final results and evaluation metrics.
    
    Args:
        results: Results to save
        args: Command line arguments
        results_dir: Directory to save results
    """
    try:
        # Clean dataset type for filename
        if "-" in args.dataset_type:
            clean_dataset_type = args.dataset_type.replace(" ", "").replace("-", "-")
        else:
            clean_dataset_type = args.dataset_type

        # Generate output file names
        base_filename = f"{args.run_name}_{clean_dataset_type}_{args.input_mode}_{args.fewshot_mode}_{args.num_examples}shots"
        if args.output_suffix:  # Empty string "" will evaluate to False
            base_filename += f"_{args.output_suffix}"
        
        # Save raw results
        results_file = os.path.join(results_dir, f"{base_filename}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        
        # Evaluate results based on dataset type
        logger.info("Evaluating results")
        all_metrics = {}
        
        if "-" in args.dataset_type:
            dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]
        else:
            dataset_types = [DatasetType(args.dataset_type)]
        
        for dt in dataset_types:
            dt_results = [r for r in results if r["dataset_type"] == dt.value]
            if dt_results:
                # Clean predictions based on dataset type
                for r in dt_results:
                    r["predicted_label (cleaned)"] = clean_prediction(r["predicted_label"], dt)
                
                metrics = evaluate_predictions(dt_results, dt)
                all_metrics[dt.value] = metrics
                logger.info(f"Results for {dt}:")
                for metric, value in metrics.items():
                    if isinstance(value, (float, int)):
                        logger.info(f"  {metric}: {value:.4f}")
                    else:
                        logger.info(f"  {metric}: {value}")
        
        # Save metrics
        metrics_file = os.path.join(results_dir, f"{base_filename}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_file}")
        
    except Exception as e:
        logger.error(f"Error saving final results: {str(e)}")
        logger.debug(traceback.format_exc())

class DummyContextManager:
    """Dummy context manager for when autocast is not used."""
    def __enter__(self): return self
    def __exit__(self, *args): pass

def main():
    """Main entry point for inference script."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Run inference
        results = run_inference(args)
        
        # Exit successfully
        logger.info("Inference completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
