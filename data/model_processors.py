import abc
import logging
import torch
from typing import Dict, List, Optional, Any
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .master_config import DatasetType

logger = logging.getLogger(__name__)

class ModelProcessor(abc.ABC):
    """Abstract base class for model-specific processing"""
    
    @abc.abstractmethod
    def process_inputs(self, 
                       data: Dict[str, Any],
                       is_training: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process inputs based on model requirements.
        
        Args:
            data: Dictionary containing all necessary data:
                - text: The main text input
                - template: The prompt template to use
                - examples: List of few-shot examples
                - fewshot_mode: Mode for few-shot examples ('text' or 'speech')
                - input_mode: Mode for input ('speech_only', 'text_only')
                - completion: Target completion (for training)
                - audio: Main audio data (if applicable)
                - examples_audio: Audio data for few-shot examples (if applicable)
            is_training: Whether this is for training (affects processing)
            
        Returns:
            Dictionary of processed inputs as tensors
        """
        pass
    
    @abc.abstractmethod
    def format_prompt(self, 
                      template: str, 
                      text: str, 
                      examples: Optional[List[Dict]] = None) -> str:
        """Format prompt according to model requirements"""
        pass
    
    @abc.abstractmethod
    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of items for model input"""
        pass

class QwenProcessor(ModelProcessor):
    """
    Processor for Qwen2 Audio model.
    Handles processing of inputs and targets for the model.
    """
    
    def __init__(self, processor, max_length=512):
        """
        Initialize the Qwen processor.
        
        Args:
            processor: The Qwen2 processor from Hugging Face
            max_length: Maximum sequence length for tokenization
        """
        self.processor = processor
        self.max_length = max_length
        self.batch_counter = 0 

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """Process inputs for Qwen2 model"""
        text = data.get("prompt", "")
        audio = data.get("audio")
        examples_audio = data.get("examples_audio")
        completion = data.get("completion", "")
        input_mode = data.get("input_mode", "speech_only")
        
        # Prepare audio inputs
        audios = []
        if examples_audio is not None:
            audios.extend(examples_audio)
        
        if audio is not None:
            audios.append(audio)
        
        # Process text input
        input_text = text
        if is_training:
            # Add completion with EOS token for training
            completion_with_eos = f"{completion}{self.processor.tokenizer.eos_token}"
            input_text = f"{text}{completion_with_eos}"
        
        # Calculate prompt length by tokenizing prompt separately
        prompt_tokens = self.processor.tokenizer(text, return_tensors="pt").input_ids
        prompt_length = prompt_tokens.size(1)
        
        # STRICT SEPARATION: Handle text_only mode separately
        if input_mode == "text_only":
            # Use tokenizer only, no audio processor
            inputs = self.processor.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            if 'input_ids' in inputs:
                try:
                    decoded = self.processor.tokenizer.decode(inputs['input_ids'][0][:50])
                    logging.info(f"TEXT_ONLY - Decoded first 50 tokens: {decoded}")
                except Exception as e:
                    logging.error(f"Error decoding tokens: {str(e)}")
            
            return {
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "prompt_length": prompt_length
                # No audio features for text_only mode
            }
        
        # STRICT SEPARATION: Handle speech_only mode
        else:
            inputs = self.processor(
                text=input_text,
                audios=audios if len(audios) > 0 else None,
                return_tensors="pt",
                sampling_rate=16000
            )
            
            if 'input_ids' in inputs:
                try:
                    decoded = self.processor.tokenizer.decode(inputs['input_ids'][0][:50])
                    logging.info(f"SPEECH_ONLY - Decoded first 50 tokens: {decoded}")
                except Exception as e:
                    logging.error(f"Error decoding tokens: {str(e)}")
            else:
                logging.warning("No input_ids in processor output!")
            
            
            # Convert to float16 for efficiency
            if hasattr(inputs, 'input_features') and inputs.input_features is not None:
                inputs.input_features = inputs.input_features.to(torch.float16)
            

            return {
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "input_features": inputs.input_features.squeeze(0) if hasattr(inputs, 'input_features') else None,
                "feature_attention_mask": inputs.feature_attention_mask.squeeze(0) if hasattr(inputs, 'feature_attention_mask') else None,
                "prompt_length": prompt_length
            }

    def format_prompt(self, 
                      template: str, 
                      text: str, 
                      examples: Optional[List[Dict]] = None,
                      input_mode: str = "speech_only",
                      fewshot_mode: str = "text",
                      dataset_type: Optional[DatasetType] = None,
                      **kwargs) -> str:
        """Format a prompt for Qwen2 model using the chat template"""
        # Create conversation format for Qwen2
        conversation = [
            {'role': 'system', 'content': template}
        ]
        
        user_content = []
        
        # Add examples if provided
        if examples and len(examples) > 0:
            user_content.append({"type": "text", "text": "Here are few examples to learn from:\n"})
            
            for example in examples:
                example_text = example.get("text", "")
                example_label = example.get("label", "")
                
                if fewshot_mode == 'speech':
                    user_content.extend([
                        {"type": "audio", "audio_url": "dummy_url"},  # Placeholder, actual audio is passed separately
                        {"type": "text", "text": f"Label: {example_label}\n"}
                    ])
                else:  # text mode
                    user_content.extend([
                        {"type": "text", "text": f"Text: {example_text}\n"},
                        {"type": "text", "text": f"Label: {example_label}\n"}
                    ])
        
        # Add current input instruction
        user_content.append({"type": "text", "text": "\nNow analyze this input:\n"})
        
        # STRICT SEPARATION: Only text_only or speech_only allowed
        if input_mode == "text_only":
            # Text only - no audio placeholders
            user_content.append({"type": "text", "text": text})
        else:
            # Speech only (Default) - audio placeholder, no transcript text
            user_content.append({"type": "audio", "audio_url": "dummy_url"})
        
        # Add user message to conversation
        conversation.append({
            "role": "user", 
            "content": user_content
        })
        
        # Apply chat template to get formatted text
        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        return formatted_prompt
    
    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of items for model input - handles text_only and speech_only modes"""
        # Initialize batch dictionary
        batch = {}
        
        # Get all keys from the first item
        keys = batch_items[0].keys()
        
        # Process each key
        for key in keys:
            if key in ["input_ids", "attention_mask"]:
                # Stack tensors
                batch[key] = torch.stack([item[key] for item in batch_items if key in item])
            elif key == "input_features":
                # Only process if items actually have input_features (not None)
                if all(item.get("input_features") is not None for item in batch_items):
                    # For input features, we need to check the shape
                    if len(batch_items[0]["input_features"].shape) == 3:  # Has examples
                        # Concatenate all input features
                        batch[key] = torch.cat([item["input_features"] for item in batch_items])
                    else:  # No examples, just stack normally
                        batch[key] = torch.stack([item["input_features"] for item in batch_items])
                # If some items don't have input_features, don't add to batch (text_only mode)
            elif key == "feature_attention_mask":
                # Only process if items actually have feature_attention_mask (not None)
                if all(item.get("feature_attention_mask") is not None for item in batch_items):
                    # For feature attention mask, we need to check the shape
                    if len(batch_items[0]["feature_attention_mask"].shape) == 2:  # Has examples
                        # Concatenate all feature attention masks
                        batch[key] = torch.cat([item["feature_attention_mask"] for item in batch_items])
                    else:  # No examples, just stack normally
                        batch[key] = torch.stack([item["feature_attention_mask"] for item in batch_items])
                # If some items don't have feature_attention_mask, don't add to batch (text_only mode)
            elif key == "prompt_length":
                # Convert to tensor
                batch[key] = torch.tensor([item["prompt_length"] for item in batch_items])
            elif key in ["prompt", "text", "true_label", "dataset_type", "completion"]:
                # Collect non-tensor data
                batch[key] = [item[key] for item in batch_items if key in item]
        
        return batch


class SalmonProcessor(ModelProcessor):
    """
    Processor for SALMONN model.
    Handles processing of inputs and targets for the model.
    """
    
    def __init__(self, tokenizer, max_length=128):
        """
        Initialize the SALMONN processor.
        
        Args:
            processor: The SALMONN processor (contains tokenizer and feature extractor)
            max_length: Maximum sequence length for tokenization
        """

        # Initialize the processor for compatibility with the ICL framework

        from transformers import WhisperFeatureExtractor
        whisper_path = "openai/whisper-large-v2"
        logging.info(f"Initializing WhisperFeatureExtractor from {whisper_path}")
        self.processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_counter = 0
    
    def process_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """Process inputs for SALMONN model"""
        prompt = data.get("prompt", "")
        audio = data.get("audio")
        examples_audio = data.get("examples_audio", [])
        completion = data.get("completion", "")
        input_mode = data.get("input_mode", "speech_only")

        # Process text input
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Process main audio if provided
        spectrogram = None
        raw_wav = None
        wav_length = 0
        
        # STRICT SEPARATION: Only process audio if input_mode is NOT text_only
        if audio is not None and input_mode != "text_only":
            raw_wav = torch.tensor(audio)  # 1D tensor
            spectrogram = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.squeeze(0)  # Remove batch dimension
            wav_length = len(raw_wav)
        
        if self.batch_counter == 0:
            logging.info(f"\n=== Input Processing Debug ===")
            logging.info(f"Input mode: {input_mode}")
            logging.info(f"Spectrogram: {'Present' if spectrogram is not None else 'None'}")
    
        # Process examples
        examples_speech = []
        if examples_audio and len(examples_audio) > 0:
            for example_audio in examples_audio:
                # Process each example same as main audio
                example_raw_wav = torch.tensor(example_audio)  # 1D tensor
                example_spectrogram = self.processor(
                    example_audio, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.squeeze(0)  # Remove batch dimension
                
                examples_speech.append({
                    "raw_wav": example_raw_wav,
                    "spectrogram": example_spectrogram,
                    "wav_length": len(example_raw_wav)
                })

        self.batch_counter += 1
        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "spectrogram": spectrogram,  # (80, 3000) or None
            "raw_wav": raw_wav,  # 1D tensor or None
            "wav_length": wav_length,
            "examples_speech": examples_speech,  # List of dicts with raw_wav, spectrogram, wav_length
            "num_examples": len(examples_speech),
            "completion": completion
        }
    
    def format_prompt(self, 
                      template: str, 
                      text: str, 
                      examples: Optional[List[Dict]] = None,
                      input_mode: str = "speech_only",
                      fewshot_mode: str = "text",
                      dataset_type: Optional[DatasetType] = None,
                      **kwargs) -> str:
        """Format prompt for SALMONN model"""
        # Format examples if provided
        examples_text = ""
        if examples and len(examples) > 0:
            if fewshot_mode == "speech":
                # Speech examples
                examples_text = "\n\n".join([
                    f"<Speech><Example{i}></Speech>\n"
                    f"Output: {example.get('label', '')}"
                    for i, example in enumerate(examples)
                ])
            else:
                # Text examples
                examples_text = "\n\n".join([
                    f"Text: {example.get('text', '')}\n"
                    f"Output: {example.get('label', '')}"
                    for example in examples
                ])
            
            examples_text = f"\nHere are few examples to learn from:\n{examples_text}\n\n"
        
        # STRICT SEPARATION: Create input section based on input mode
        if input_mode == "text_only":
            # Text Only
            input_section = f"Text: {text}"
        else: 
            # Speech Only (Default) - No transcript text allowed
            input_section = "<Speech><SpeechHere></Speech>"
        
        # Create the final prompt
        prompt = f"{template}\n{examples_text}Now analyze this input:\n{input_section}\nOutput:"
        
        return prompt
    
    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch for SALMONN model"""
        batch = {}
        
        # Process text inputs
        batch["input_ids"] = torch.stack([item["input_ids"] for item in batch_items])
        batch["attention_mask"] = torch.stack([item["attention_mask"] for item in batch_items])
        
        # Check for valid speech data
        has_valid_speech = all(item.get('spectrogram') is not None for item in batch_items)
        
        if has_valid_speech:
            # Only add speech-related tensors if we actually have speech
            wav_lengths = torch.tensor([item['wav_length'] for item in batch_items])
            raw_wavs = [item['raw_wav'] for item in batch_items]
            raw_wavs_padded = pad_sequence(raw_wavs, batch_first=True, padding_value=0)
            padding_mask = torch.arange(raw_wavs_padded.size(1)).unsqueeze(0) >= wav_lengths.unsqueeze(1)
            spectrograms = torch.stack([item['spectrogram'] for item in batch_items])
            
            batch["wav_lengths"] = wav_lengths
            batch["raw_wav"] = raw_wavs_padded
            batch["padding_mask"] = padding_mask
            batch["spectrogram"] = spectrograms
        
        # Process examples
        max_examples = max(item['num_examples'] for item in batch_items)
        if max_examples > 0:
            has_speech_examples = any(
                'examples_speech' in item and 
                len(item['examples_speech']) > 0 and
                item['examples_speech'][0]['spectrogram'] is not None
                for item in batch_items
            )
            
            if has_speech_examples:
                # Find max length for padding
                max_length = max(
                    example['wav_length']
                    for item in batch_items if 'examples_speech' in item
                    for example in item['examples_speech'][:item['num_examples']]
                )
                
                example_specs = []
                example_wavs = []
                example_masks = []
                
                for item in batch_items:
                    examples = item.get('examples_speech', [])[:item['num_examples']]
                    batch_specs = []
                    batch_wavs = []
                    batch_masks = []
                    
                    for example in examples:
                        spec = example['spectrogram']
                        wav = example['raw_wav']
                        wav_length = example['wav_length']
                        
                        batch_specs.append(spec)
                        
                        if wav.size(0) < max_length:
                            wav = F.pad(wav, (0, max_length - wav.size(0)), value=0)
                        batch_wavs.append(wav)
                        
                        mask = torch.arange(max_length, device=wav.device) >= wav_length
                        batch_masks.append(mask)
                    
                    # Pad to max_examples
                    while len(batch_specs) < max_examples:
                        pad_spec = torch.zeros_like(batch_specs[0]) if batch_specs else torch.zeros((80, 3000))
                        batch_specs.append(pad_spec)
                        batch_wavs.append(torch.zeros(max_length, device=wav.device))
                        batch_masks.append(torch.ones(max_length, device=wav.device, dtype=torch.bool))
                    
                    example_specs.append(torch.stack(batch_specs))
                    example_wavs.append(torch.stack(batch_wavs))
                    example_masks.append(torch.stack(batch_masks))
                
                batch["example_spectrograms"] = torch.stack(example_specs)
                batch["example_wavs"] = torch.stack(example_wavs)
                batch["example_padding_masks"] = torch.stack(example_masks)

        
        # Add non-tensor data
        batch["num_examples"] = torch.tensor([item["num_examples"] for item in batch_items])
        for key in ["prompt", "completion", "text", "dataset_type"]:
            if key in batch_items[0]:
                batch[key] = [item[key] for item in batch_items]
        
        return batch


def get_processor(model_type: str, processor=None, tokenizer=None) -> ModelProcessor:
    """
    Factory function to get the appropriate processor for a model type.
    
    Args:
        model_type: Type of model ('salmonn', 'qwen2', etc.)
        processor: The model's processor object
        
    Returns:
        An instance of a ModelProcessor subclass
    """
    model_type = model_type.lower()
    
    if model_type == "salmonn":
        return SalmonProcessor(tokenizer)
    elif model_type in ["qwen", "qwen2"]:
        return QwenProcessor(processor)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")