import torch
from datasets import load_from_disk
from transformers import WhisperFeatureExtractor
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import json
import os
from custom_salmon import CustomSALMONN
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
# Model paths
parser.add_argument("--base_model_path", type=str, default="lmsys/vicuna-13b-v1.1")
parser.add_argument("--whisper_path", type=str, default="openai/whisper-large-v2")
parser.add_argument("--beats_path", type=str, default="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
parser.add_argument("--peft_model_path", type=str, default="", help="Path to SALMONN checkpoint")
parser.add_argument("--test_data_path", type=str, default="/path/to/test/data")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_examples", type=int, default=0, help="Number of few-shot examples (0-10)")
parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output files")
parser.add_argument('--log_file', type=str, default='inference_salmon.log')
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--device_8bit", type=int, default=0)
parser.add_argument("--input_mode", type=str, default='speech_and_text', 
                   choices=['speech_only', 'speech_and_text'],
                   help="Whether to use speech only or both speech and text")
args = parser.parse_args()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(args.log_file, mode='w'),
        logging.StreamHandler()
    ]
)

class InferenceDataset(Dataset):
    def __init__(self, dataset, wav_processor, num_examples=0, input_mode='speech_only'):
        self.dataset = dataset
        self.wav_processor = wav_processor
        self.num_examples = min(num_examples, 10)
        self.input_mode = input_mode
        if input_mode not in ['speech_only', 'speech_and_text']:
            raise ValueError("input_mode must be either 'speech_only' or 'speech_and_text'")

    def __len__(self):
        return len(self.dataset)

    def _create_prompt(self, item):
        base_prompt = """You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: Positive, Negative, or Neutral.

Guidelines:
- Choose Positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose Negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose Neutral ONLY IF the statement is purely factual with zero emotional content"""

        if 'few_shot_examples' in item and len(item['few_shot_examples']) > 0 and self.num_examples > 0:
            selected_examples = item['few_shot_examples'][:self.num_examples]
            examples_text = "\n\n".join([
                f"Text: {example['text']}\n"
                f"Output: {example['label']}"
                for example in selected_examples
            ])
            
            if self.input_mode == 'speech_only':
                input_section = "<Speech><SpeechHere></Speech>"
            else:  # speech_and_text
                input_section = f"<Speech><SpeechHere></Speech>\nTranscript: {item['normalized_text']}"
            
            prompt = f"""{base_prompt}

Here are few examples to learn from:
{examples_text}

Now analyze this input:
{input_section}
Output:"""
        else:
            if self.input_mode == 'speech_only':
                input_section = "<Speech><SpeechHere></Speech>"
            else:  # speech_and_text
                input_section = f"<Speech><SpeechHere></Speech>\nTranscript: {item['normalized_text']}"
            
            prompt = f"""{base_prompt}

Now analyze this input:
{input_section}
Output:"""
        
        return prompt

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Process audio features
        raw_wav = torch.from_numpy(item["audio"]['array'])
        spectrogram = self.wav_processor(raw_wav, sampling_rate=16000, return_tensors="pt")["input_features"]
        
        return {
            "spectrogram": spectrogram.squeeze(0),
            "raw_wav": raw_wav,
            "wav_length": len(raw_wav),
            "text": item["normalized_text"],
            "prompt": self._create_prompt(item),
            "true_sentiment": item['sentiment']
        }

def collate_fn(batch):
    wav_lengths = torch.tensor([item['wav_length'] for item in batch])
    raw_wavs = [item['raw_wav'] for item in batch]
    raw_wavs_padded = pad_sequence(raw_wavs, batch_first=True, padding_value=0)
    padding_mask = torch.arange(raw_wavs_padded.size(1)).unsqueeze(0) >= wav_lengths.unsqueeze(1)
    spectrograms = torch.stack([item['spectrogram'] for item in batch])
    
    return {
        "spectrogram": spectrograms,
        "raw_wav": raw_wavs_padded,
        "padding_mask": padding_mask,
        "prompt": [item['prompt'] for item in batch],
        "true_sentiment": [item['true_sentiment'] for item in batch],
        "text": [item['text'] for item in batch]
    }

def evaluate_results(predictions, output_file):
    df = pd.DataFrame(predictions)
    df.columns = ['text', 'gt', 'pd']
    
    valid_classes = ['neutral', 'negative', 'positive']
    total_samples = len(df)
    
    df = df[df['gt'].str.lower().isin(valid_classes)]
    after_gt_filter = len(df)
    
    df = df[df['pd'].str.lower().isin(valid_classes)]
    after_pred_filter = len(df)
    
    matrix = confusion_matrix(
        df['gt'].str.lower().values,
        df['pd'].str.lower().values,
        labels=valid_classes
    )
    
    class_accuracy = matrix.diagonal()/matrix.sum(axis=1)
    class_samples = matrix.sum(axis=1)
    
    macro_f1 = f1_score(
        df['gt'].str.lower().values,
        df['pd'].str.lower().values,
        average="macro",
        labels=valid_classes
    )
    
    # Save detailed metrics
    with open(output_file, "w") as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Valid ground truth samples: {after_gt_filter}\n")
        f.write(f"Valid prediction samples: {after_pred_filter}\n")
        f.write(f"\nMacro F1 Score: {macro_f1:.4f}\n")
        f.write("\nPer-class metrics:\n")
        for i, class_name in enumerate(valid_classes):
            f.write(f"{class_name}:\n")
            f.write(f"  Accuracy: {class_accuracy[i]:.4f}\n")
            f.write(f"  Samples: {class_samples[i]}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"Classes: {valid_classes}\n")
        f.write(str(matrix))
    
    return {
        'macro_f1': macro_f1,
        'class_accuracy': class_accuracy,
        'confusion_matrix': matrix,
        'valid_samples': after_pred_filter,
        'filtered_samples': total_samples - after_pred_filter
    }



def load_model_weights(model, base_checkpoint_path, finetuned_checkpoint_path=None, device='cpu'):
    """
    Load model weights with optional finetuned weights
    Args:
        model: The SALMONN model
        base_checkpoint_path: Path to original SALMONN weights
        finetuned_checkpoint_path: Optional path to finetuned weights (if empty string, only base weights used)
        device: Device to load the model on
    """
    logging.info(f"Loading base weights from {base_checkpoint_path}")
    base_checkpoint = torch.load(base_checkpoint_path, map_location='cpu')
    base_state_dict = base_checkpoint['model'] if 'model' in base_checkpoint else base_checkpoint
    
    # Load base weights
    missing_keys, unexpected_keys = model.load_state_dict(base_state_dict, strict=False)
    logging.info(f"Base model loading - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    # Load finetuned weights if path is provided and not empty
    if finetuned_checkpoint_path and finetuned_checkpoint_path.strip():
        logging.info(f"Loading finetuned weights from {finetuned_checkpoint_path}")
        try:
            finetuned_checkpoint = torch.load(finetuned_checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in finetuned_checkpoint:
                finetuned_state_dict = finetuned_checkpoint['model_state_dict']
                logging.info("Found 'model_state_dict' in checkpoint")
            elif 'model' in finetuned_checkpoint:
                finetuned_state_dict = finetuned_checkpoint['model']
                logging.info("Found 'model' in checkpoint")
            else:
                raise KeyError(f"No model state dict found in checkpoint. Keys found: {finetuned_checkpoint.keys()}")

            # Log which keys will be updated
            finetuned_keys = set(finetuned_state_dict.keys())
            base_keys = set(base_state_dict.keys())
            logging.info(f"Updating {len(finetuned_keys)} parameters from finetuned model")
            if finetuned_keys - base_keys:
                logging.info(f"New keys in finetuned model: {finetuned_keys - base_keys}")
            
            # Update only the finetuned parameters
            model.load_state_dict(finetuned_state_dict, strict=False)
            logging.info("Successfully loaded finetuned weights")
        except Exception as e:
            logging.error(f"Error loading finetuned weights: {str(e)}")
            logging.warning("Continuing with base weights only")
    else:
        logging.info("No finetuned weights path provided, using base weights only")
    
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize model
    logging.info("Loading SALMONN model...")
    logging.info(f"Loading checkpoint from: {args.peft_model_path}")
    
    config = {
        "llama_path": args.base_model_path,
        "whisper_path": args.whisper_path,
        "beats_path": args.beats_path,
        "use_speech_Qformer": True,
        "freeze_whisper": True,
        "freeze_beats": True,
        "freeze_speech_QFormer": True,
        "num_speech_query_token": 1,
        "window_level_Qformer": True,
        "second_per_window": 0.333333,
        "second_stride": 0.333333,
        "low_resource": args.load_in_8bit,
        "device_8bit": args.device_8bit,
    }
    
    model = CustomSALMONN.from_config(config)
    
    # Load weights using existing peft_model_path argument
    model = load_model_weights(
        model=model,
        base_checkpoint_path="/home/sriramg/aneeraj/storage/salmonn_v1.pth",
        finetuned_checkpoint_path=args.peft_model_path,  # If empty string, only base weights will be used
        device=device
    )
    
    # Initialize wav processor
    wav_processor = WhisperFeatureExtractor.from_pretrained(args.whisper_path)
    
    # Load dataset
    logging.info("Loading test dataset...")
    test_dataset = load_from_disk(args.test_data_path)
    test_dataset = InferenceDataset(
        test_dataset, 
        wav_processor, 
        num_examples=args.num_examples,
        input_mode=args.input_mode
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    results = {'predictions': []}
    
    logging.info("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # Move inputs to device
            batch["spectrogram"] = batch["spectrogram"].to(device)
            batch["raw_wav"] = batch["raw_wav"].to(device)
            batch["padding_mask"] = batch["padding_mask"].to(device)
            
            # Generate predictions using the model's generate_sentiment method
            predictions = model.generate_sentiment(batch)
            
            # Store results
            for pred, true_sentiment, text in zip(predictions, batch["true_sentiment"], batch["text"]):
                print(f"\nTrue: {true_sentiment.lower()}, Predicted: {pred.lower()}", flush=True)
                results['predictions'].append({
                    'text': text,
                    'true_sentiment': true_sentiment.lower(),
                    'predicted_sentiment': pred.lower()
                })
    
    # Save results
    base_filename = f'salmon_{args.output_suffix}_{args.num_examples}shot'
    os.makedirs('./results/final_results', exist_ok=True)
    
    # Save raw predictions
    with open(f'./results/final_results/{base_filename}_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Evaluate and save metrics
    metrics = evaluate_results(
        [[p['text'], p['true_sentiment'], p['predicted_sentiment']] for p in results['predictions']],
        f'./results/final_results/{base_filename}_metrics.txt'
    )
    
    # Log summary
    logging.info(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    logging.info(f"Valid samples: {metrics['valid_samples']}")
    logging.info(f"Filtered samples: {metrics['filtered_samples']}")
    logging.info(f"Results saved to {base_filename}_*")

if __name__ == "__main__":
    main() 