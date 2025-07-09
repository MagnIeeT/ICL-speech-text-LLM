import torch
from datasets import load_from_disk
from transformers import WhisperFeatureExtractor
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import json
import os
from dataset_config import DatasetType, DATASET_CONFIGS, DatasetSplit
from salmon_datasets import InferenceDataset, collate_fn
from evaluation_utils import evaluate_results
import time
from typing import List, Dict
import ast
from utils.generate_fewshots import convert_ner_to_dict
from peft import LoraConfig, get_peft_model,TaskType

parser = argparse.ArgumentParser()
# Model paths
parser.add_argument("--base_model_path", type=str, 
                   default="lmsys/vicuna-13b-v1.1",  # Update default path
                   help="Path to SALMONN base checkpoint")
parser.add_argument("--whisper_path", type=str, default="openai/whisper-large-v2")
parser.add_argument("--beats_path", type=str, default="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
parser.add_argument("--peft_model_path", type=str, default="", help="Path to SALMONN checkpoint")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_examples", type=int, default=0, help="Number of few-shot examples (0-10)")
parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output files")
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--device_8bit", type=int, default=0)
parser.add_argument("--input_mode", type=str, default='speech_and_text', 
                   choices=['speech_only', 'speech_and_text','text_only'],
                   help="Whether to use speech only or both speech and text")
parser.add_argument("--dataset_type", type=str, choices=["voxceleb", "hvb", "voxpopuli"], 
                   default="voxceleb", help="Type of dataset to use")
parser.add_argument("--fewshot_mode", type=str, default='text', 
                   choices=['text', 'speech'],
                   help="Whether to use text or speech for few-shot examples")
parser.add_argument("--run_name", type=str, required=True, help="Name of the model/run")
parser.add_argument("--today", type=str, help="Current date for organizing outputs")
parser.add_argument("--model_type", type=str, 
                   choices=["salmonn", "qwen2"],
                   default="salmonn", 
                   help="Type of model to use")

args = parser.parse_args()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def load_model_weights(model, base_checkpoint_path, finetuned_checkpoint_path=None, device='cpu'):
    """
    Load model weights with explicit dtype handling
    """
    # Load base weights
    logging.info(f"Loading base weights from {base_checkpoint_path}")
    base_checkpoint = torch.load(base_checkpoint_path, map_location=device)
    base_state_dict = base_checkpoint['model']  # Define base_state_dict here
    
    # Check if weights match model architecture
    missing, unexpected = model.load_state_dict(base_state_dict, strict=False)
    logging.info(f"Base model loading - Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    # Load finetuned weights if provided
    if finetuned_checkpoint_path and finetuned_checkpoint_path.strip():
        logging.info(f"Loading finetuned weights from {finetuned_checkpoint_path}")
        finetuned_checkpoint = torch.load(finetuned_checkpoint_path, map_location=device)
        finetuned_state_dict = finetuned_checkpoint['model'] 

        finetuned_keys = set(finetuned_state_dict.keys())
        base_keys = set(base_state_dict.keys())  # Now base_state_dict is defined
        logging.info(f"Updating {len(finetuned_keys)} parameters from finetuned model")
        if finetuned_keys - base_keys:
            logging.info(f"New keys in finetuned model: {finetuned_keys - base_keys}")
        
        # Update only the finetuned parameters
        model.load_state_dict(finetuned_state_dict, strict=False)

        logging.info("Successfully loaded finetuned weights")

    model.to(device)
    model.eval()
    return model

def clean_prediction(prediction: str, dataset_type: str = None) -> str:
    """Clean prediction by removing escape characters and dataset-specific artifacts"""
    # First remove basic escape characters
    cleaned = prediction.replace('\\', '')

    if '\n' in cleaned:
        cleaned = cleaned.split('\n')[0]
    
    return cleaned.strip()

def generate_output(model, processor, batch, model_type):
    """Generate output based on model type"""
    if model_type == "qwen2":
        # Debug shapes

        with torch.cuda.amp.autocast(dtype=torch.float16):
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                input_features=batch["input_features"],
                feature_attention_mask=batch["feature_attention_mask"],
                max_new_tokens=10,
            )
            
            new_tokens = generated_ids[:, batch["input_ids"].size(1):]
            generated_ids = generated_ids[:, batch["input_ids"].size(1):]
            
            # Decode the generated IDs
            outputs = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return outputs
    else:
        # Existing SALMONN generation
        return model.generate_output(batch, input_mode=args.input_mode)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if args.model_type == "qwen2":
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        # logging.info(f"Loading Qwen2 model from {args.base_model_path}")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16  # Force model to float16
        )
        
        # Load processor for audio and text processing
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
        )
        model.processor = processor  # Attach processor to model for convenience
        
        # Load finetuned weights if provided
        if args.peft_model_path and args.peft_model_path.strip():

            lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.1,
            inference_mode=False, 
            task_type=TaskType.CAUSAL_LM,
        )

            model = get_peft_model(model, lora_config)

            logging.info(f"Loading finetuned weights for Qwen2 from {args.peft_model_path}")
            checkpoint = torch.load(args.peft_model_path, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=False)
            logging.info("Successfully loaded finetuned weights")
        
        model.eval()
        wav_processor = processor  # Use Qwen2's processor instead of Whisper
    else:
        # Original SALMONN initialization
        from custom_salmon import CustomSALMONN
        config = {
            "llama_path": args.base_model_path,
            "whisper_path": args.whisper_path,
            "beats_path": args.beats_path,
            "use_speech_Qformer": True,
            "freeze_whisper": True,
            "freeze_beats": True,
            "freeze_speech_QFormer": False,
            "num_speech_query_token": 1,
            "window_level_Qformer": True,
            "second_per_window": 0.333333,
            "second_stride": 0.333333,
            # "low_resource": args.load_in_8bit,
            "low_resource": True,
        }
        
        model = CustomSALMONN.from_config(config)
        

        model = load_model_weights(model, "/home/sriramg/aneeraj/storage/salmonn_v1.pth", args.peft_model_path, device)
        model.eval()
        wav_processor = WhisperFeatureExtractor.from_pretrained(args.whisper_path)
        
    
    # Load dataset
    dataset_load_start = time.time()
    dataset_type = DatasetType(args.dataset_type)
    dataset_config = DATASET_CONFIGS[dataset_type]
    test_path = dataset_config.get_path(DatasetSplit.TEST)
    # test_path = dataset_config.get_path(DatasetSplit.TRAIN)
    
    logging.info(f"Loading dataset from: {test_path}")
    dataset = load_from_disk(test_path)
    # dataset= dataset.select(range(20))
    logging.info(f"Dataset loading took {time.time() - dataset_load_start:.2f} seconds")
    
    # Create inference dataset
    inference_dataset_start = time.time()
    test_dataset = InferenceDataset(
        dataset_type=dataset_type,
        dataset=dataset,
        wav_processor=wav_processor,
        num_examples=args.num_examples,
        input_mode=args.input_mode,
        fewshot_mode=args.fewshot_mode,
        model_type=args.model_type,
        run_name=args.run_name
    )
    logging.info(f"Inference dataset creation took {time.time() - inference_dataset_start:.2f} seconds")
    
    if args.fewshot_mode == 'speech':
        audio_lookup_path = dataset_config.get_audio_lookup_path(DatasetSplit.TEST)
        logging.info(f"Using audio lookup from: {audio_lookup_path}")
    
    # Create dataloader
    dataloader_start = time.time()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    logging.info(f"Dataloader creation took {time.time() - dataloader_start:.2f} seconds")
    
    # Initialize model

    # Initialize predictions list
    predictions = []
    
    # Add timing for the batch processing
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Running inference")):
            batch_start = time.time()
            
            try:
                # Move batch tensors to device based on model type
                device_start = time.time()
                if args.model_type == "qwen2":
                    # Move all tensors to device and convert audio features to float16
                    batch["input_ids"] = batch["input_ids"].to(device)
                    batch["attention_mask"] = batch["attention_mask"].to(device)
                    batch["input_features"] = batch["input_features"].to(device).to(torch.float16)  # Explicit float16
                    batch["feature_attention_mask"] = batch["feature_attention_mask"].to(device)
                else:
                    # For SALMONN, move spectrogram, raw_wav, and padding_mask
                    batch["spectrogram"] = batch["spectrogram"].to(device)
                    if batch["raw_wav"] is not None:
                        batch["raw_wav"] = batch["raw_wav"].to(device)
                    if batch["padding_mask"] is not None:
                        batch["padding_mask"] = batch["padding_mask"].to(device)
                device_time = time.time() - device_start

                if batch_idx == 0:
                    logging.info(f"Prompt example:\n{batch['prompt'][0]}")
                    logging.info(f"label:\n{batch['completion'][0]}")
                
                # Generate outputs
                if args.model_type == "qwen2":
                    if batch_idx == 0:
                        logging.info(f"input_features shape: {batch['input_features'].shape}")
                        logging.info(f"input_features_mask shape: {batch['feature_attention_mask'].shape}")

                    outputs = generate_output(model, wav_processor, batch, args.model_type)
                else:
                    outputs = model.generate_output(batch, input_mode=args.input_mode)
                
                # Process outputs
                for i, (output, true_sentiment) in enumerate(zip(outputs, batch["completion"])):
                    cleaned_output = clean_prediction(output, args.dataset_type)
                    logging.info(f"Predicted (original): {output}")
                    logging.info(f"Predicted (cleaned): {cleaned_output}")
                    logging.info(f"True: {true_sentiment}")
                    logging.info("-" * 50)
                    
                    # Create prediction entry
    
                    prediction = {
                        "text": batch["text"][i],
                        "true_sentiment": true_sentiment,  # Already in correct format from dataset
                        "predicted_sentiment": cleaned_output,
                        "predicted_sentiment_org": output.strip()
                    }
                    
                    predictions.append(prediction)
                
                batch_time = time.time() - batch_start
                if batch_idx == 0:
                    logging.info(f"Batch {batch_idx + 1} - Total time: {batch_time:.2f}s (Device transfer: {device_time:.2f}s)")

                # Clear memory every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    
                    logging.info(f"Cleared memory at batch {batch_idx + 1}")
                    
                    # Log memory status
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
                        logging.info(f"GPU Memory - Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"OOM error at batch {batch_idx}. Attempting to recover...")
                    # Clear memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    # Create output filenames with proper directory
    os.makedirs(f"/data2/neeraja/neeraja/results/model_ICL/metrics/{args.today}", exist_ok=True)
    base_filename = f"{args.dataset_type}_{args.run_name}_{args.input_mode}_{args.fewshot_mode}_{args.num_examples}shots"
    predictions_file = f"/data2/neeraja/neeraja/results/model_ICL/metrics/{args.today}/{base_filename}_predictions.json"
    metrics_file = f"/data2/neeraja/neeraja/results/model_ICL/metrics/{args.today}/{base_filename}_metrics.txt"
    
    # Save full predictions with all keys
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Convert predictions to format expected by evaluate_results (only 3 keys)
    eval_predictions = [
        {
            'text': p['text'],
            'gt': p['true_sentiment'],
            'pd': p['predicted_sentiment']
        } for p in predictions
    ]
    
    # Evaluate results and save metrics
    evaluate_results(eval_predictions, metrics_file, args.dataset_type)

if __name__ == "__main__":
    main() 


# input_ids shape: torch.Size([1, 441])
# attention_mask shape: torch.Size([1, 441])
# input_features shape: torch.Size([1, 128, 3000])
# feature_attention_mask shape: torch.Size([1, 3000]),