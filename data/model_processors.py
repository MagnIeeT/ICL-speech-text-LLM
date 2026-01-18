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
                - input_mode: Mode for input ('speech_only', 'text_only', 'speech_and_text')
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
        """Add dataset type routing"""
        dataset_type = data.get("dataset_type")
        
        if dataset_type == DatasetType.SQA:
            return self._process_sqa_inputs(data, is_training)
        else:
            return self._process_default_inputs(data, is_training)
        
    def _process_sqa_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """Process SQA dataset inputs for Qwen2 model."""
        text = data.get("prompt", "")
        audio_data = data.get("audio", {})
        examples_audio = data.get("examples_audio", [])
        completion = data.get("completion", "")
        input_mode = data.get("input_mode", "text_only")
        # Prepare audio inputs
        audios = []
        
        # Add example audios if available
        if examples_audio:
            for example in examples_audio:
                if example.get("question_audio") is not None:
                    audios.append(example["question_audio"])
                if example.get("document_audio") is not None:
                    audios.append(example["document_audio"])
        
        # Add current question and document audio
        if audio_data.get("question_audio") is not None:
            audios.append(audio_data["question_audio"])
        if audio_data.get("document_audio") is not None:
            audios.append(audio_data["document_audio"])
        
        # Process text input
        input_text = text
        if is_training:
            # Add completion with EOS token for training
            completion_with_eos = f"{completion}{self.processor.tokenizer.eos_token}"
            input_text = f"{text}{completion_with_eos}"
        
        # Calculate prompt length by tokenizing prompt separately
        prompt_tokens = self.processor.tokenizer(text, return_tensors="pt").input_ids
        prompt_length = prompt_tokens.size(1)
        
        # Process inputs with the Qwen2 processor
        inputs = self.processor(
            text=input_text,
            audios=audios,
            return_tensors="pt",
            sampling_rate=16000
        )
        
        if 'input_ids' in inputs:
            try:
                decoded = self.processor.tokenizer.decode(inputs['input_ids'][0][:50])
                logging.info(f"SQA Decoded first 50 tokens: {decoded}")
            except Exception as e:
                logging.error(f"Error decoding SQA tokens: {str(e)}")
        else:
            logging.warning("No input_ids in SQA processor output!")
        
        # Convert to float16 for efficiency
        inputs.input_features = inputs.input_features.to(torch.float16)


        if input_mode == "text_only":
            return {
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "prompt_length": prompt_length
                # Intentionally omitting input_features and feature_attention_mask
            }
        else:
            # Convert to float16 for efficiency
            inputs.input_features = inputs.input_features.to(torch.float16)
            
            return {
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "input_features": inputs.input_features.squeeze(0),
                "feature_attention_mask": inputs.feature_attention_mask.squeeze(0),
                "prompt_length": prompt_length
            }
    
    def _process_default_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """
        Process inputs for Qwen2 model.
        
        Args:
            data: Dictionary containing all necessary data:
                - text: The main text input (formatted prompt)
                - template: The prompt template to use
                - examples: List of few-shot examples
                - fewshot_mode: Mode for few-shot examples ('text' or 'speech')
                - input_mode: Mode for input ('speech_only', 'text_only', 'speech_and_text')
                - completion: Target completion (for training)
                - audio: Main audio data (if applicable)
                - examples_audio: Audio data for few-shot examples (if applicable)
            is_training: Whether this is for training (affects processing)
            
        Returns:
            Dictionary of processed inputs as tensors
        """
        text = data.get("prompt", "")
        audio = data.get("audio")
        examples_audio = data.get("examples_audio")
        completion = data.get("completion", "")
        
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
        
        inputs = self.processor(
            text=input_text,
            audios=audios,
            return_tensors="pt",
            sampling_rate=16000
        )
        
        if 'input_ids' in inputs:
            try:
                decoded = self.processor.tokenizer.decode(inputs['input_ids'][0][:50])
                logging.info(f"Decoded first 50 tokens: {decoded}")
            except Exception as e:
                logging.error(f"Error decoding tokens: {str(e)}")
        else:
            logging.warning("No input_ids in processor output!")
        
        
        # Convert to float16 for efficiency
        inputs.input_features = inputs.input_features.to(torch.float16)
        

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "input_features": inputs.input_features.squeeze(0),
            "feature_attention_mask": inputs.feature_attention_mask.squeeze(0),
            "prompt_length": prompt_length
        }
    

    def format_prompt(self, 
                     template: str, 
                     text: str, 
                     examples: Optional[List[Dict]] = None,
                     input_mode: str = "speech_and_text",
                     fewshot_mode: str = "text",
                     dataset_type: Optional[DatasetType] = None,
                     **kwargs) -> str:
        """Add dataset type routing for prompt formatting"""
        if dataset_type == DatasetType.SQA:
            return self._format_sqa_prompt(template, text, examples, input_mode, fewshot_mode, **kwargs)
        else:
            return self._format_default_prompt(template, text, examples, input_mode, fewshot_mode)

    def _format_sqa_prompt(self, template: str, text: str, examples: Optional[List[Dict]], 
                          input_mode: str, fewshot_mode: str, **kwargs):
        """New method for SQA prompt formatting"""
        question = kwargs.get('question', '')
        
        # Create conversation format for Qwen2
        conversation = [
            {'role': 'system', 'content': template}
        ]
        
        user_content = []
        
        # Add examples if provided
        if examples and len(examples) > 0:
            user_content.append({"type": "text", "text": "Here are few examples to learn from:\n"})
            
            for i, example in enumerate(examples):
                example_question = example.get("question", "")
                example_document = example.get("document", "")
                example_answer = example.get("answer", "")
                
                if fewshot_mode == 'speech':
                    # Add question audio placeholder
                    user_content.append({"type": "audio", "audio_url": f"question_{i}"})  # Placeholder
                    
                    # Add document audio placeholder
                    user_content.append({"type": "audio", "audio_url": f"document_{i}"})  # Placeholder
                    
                    # Add answer text
                    user_content.append({"type": "text", "text": f"Answer: {example_answer}\n"})
                else:  # text mode
                    # Add text-only example
                    user_content.extend([
                        {"type": "text", "text": f"Question: {example_question}\n"},
                        {"type": "text", "text": f"Document: {example_document}\n"},
                        {"type": "text", "text": f"Answer: {example_answer}\n"}
                    ])
        
        # Add current input instruction
        user_content.append({"type": "text", "text": "\nNow analyze this input:\n"})
        
        # Add current input based on input mode
        if input_mode == "speech_and_text" or input_mode == "speech_only":
            # Add question audio placeholder
            user_content.append({"type": "audio", "audio_url": "question"})
            
            # Add question text if in speech_and_text mode
            if input_mode == "speech_and_text" and question:
                user_content.append({"type": "text", "text": f"Question text: {question}\n"})
            
            # Add document audio placeholder
            user_content.append({"type": "audio", "audio_url": "document"})
            
            # Add document text if in speech_and_text mode
            if input_mode == "speech_and_text" and text:
                user_content.append({"type": "text", "text": f"Document text: {text}"})
        else:  # text_only mode
            # Add text-only input
            user_content.extend([
                {"type": "text", "text": f"Question: {question}\n"},
                {"type": "text", "text": f"Document: {text}"}
            ])
        
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

    def _format_default_prompt(self, template: str, text: str, examples: Optional[List[Dict]], 
                             input_mode: str, fewshot_mode: str):
        """
        Format a prompt for Qwen2 model using the chat template.
        
        Args:
            template: The prompt template to use (system message)
            text: The main text input
            examples: List of few-shot examples (already formatted)
            fewshot_mode: Mode for few-shot examples ('text' or 'speech')
            
        Returns:
            Formatted prompt string
        """
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
        
        # Add current input based on whether text is provided
        # For speech input, add audio placeholder (actual audio is passed separately)
        if input_mode == "speech_only" or input_mode == "speech_and_text":
            user_content.append({"type": "audio", "audio_url": "dummy_url"})
        
        # Add text if provided (for text_only or speech_and_text modes)
        if input_mode == "speech_and_text" and text:
            user_content.append({"type": "text", "text": text})
        
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
        """Add dataset type routing for batch collation"""
        if batch_items[0].get("dataset_type") == DatasetType.SQA:
            return self._collate_sqa_batch(batch_items)
        else:
            return self._collate_default_batch(batch_items)

    def _collate_sqa_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """New method for SQA batch collation"""
        # Implementation similar to SalmonProcessor's _collate_sqa_batch
        batch = {}
        
        # Get all keys from the first item
        keys = batch_items[0].keys()
        
        # Process each key
        for key in keys:
            if key in ["input_ids", "attention_mask"]:
                # Stack tensors
                batch[key] = torch.stack([item[key] for item in batch_items if key in item])
            elif key == "input_features" and all(item.get("input_features") is not None for item in batch_items):
                # For input features, we need to check the shape
                if len(batch_items[0]["input_features"].shape) == 3:  # Has examples
                    # Concatenate all input features
                    batch[key] = torch.cat([item["input_features"] for item in batch_items])
                else:  # No examples, just stack normally
                    batch[key] = torch.stack([item["input_features"] for item in batch_items])
            elif key == "feature_attention_mask" and all(item.get("feature_attention_mask") is not None for item in batch_items):
                # For feature attention mask, we need to check the shape
                if len(batch_items[0]["feature_attention_mask"].shape) == 2:  # Has examples
                    # Concatenate all feature attention masks
                    batch[key] = torch.cat([item["feature_attention_mask"] for item in batch_items])
                else:  # No examples, just stack normally
                    batch[key] = torch.stack([item["feature_attention_mask"] for item in batch_items])
            elif key == "prompt_length":
                # Convert to tensor
                batch[key] = torch.tensor([item["prompt_length"] for item in batch_items])
            elif key in ["prompt", "text", "true_label", "question", "dataset_type", "completion"]:
                # Collect non-tensor data
                batch[key] = [item[key] for item in batch_items if key in item]
        
        return batch

    def _collate_default_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rename existing collate_batch to _collate_default_batch"""
        # Current implementation remains the same
        """
        Collate a batch of items for model input.
        
        Args:
            batch_items: List of dictionaries with processed inputs
            
        Returns:
            Collated batch as a dictionary
        """
        # Initialize batch dictionary
        batch = {}
        
        # Get all keys from the first item
        keys = batch_items[0].keys()
        
        # Process each key
        for key in keys:
            if key in ["input_ids", "attention_mask"]:
                # Stack tensors
                batch[key] = torch.stack([item[key] for item in batch_items if key in item])
            elif key == "input_features" and all(item.get("input_features") is not None for item in batch_items):
                # For input features, we need to check the shape
                if len(batch_items[0]["input_features"].shape) == 3:  # Has examples
                    # Concatenate all input features
                    batch[key] = torch.cat([item["input_features"] for item in batch_items])
                else:  # No examples, just stack normally
                    batch[key] = torch.stack([item["input_features"] for item in batch_items])
            elif key == "feature_attention_mask" and all(item.get("feature_attention_mask") is not None for item in batch_items):
                # For feature attention mask, we need to check the shape
                if len(batch_items[0]["feature_attention_mask"].shape) == 2:  # Has examples
                    # Concatenate all feature attention masks
                    batch[key] = torch.cat([item["feature_attention_mask"] for item in batch_items])
                else:  # No examples, just stack normally
                    batch[key] = torch.stack([item["feature_attention_mask"] for item in batch_items])
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
        dataset_type = data.get("dataset_type")
        
        if dataset_type == DatasetType.SQA:
            return self._process_sqa_inputs(data, is_training)
        else:
            # Use default for both VP-NEL and other datasets
            return self._process_default_inputs(data, is_training)

    def _process_sqa_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """Process inputs for SQA dataset"""
        prompt = data.get("prompt", "")
        audio_data = data.get("audio", {})  # Dictionary containing question_audio and document_audio
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

        # Process question and document audio
        question_spectrogram = None
        question_raw_wav = None
        question_wav_length = 0
        
        document_spectrogram = None
        document_raw_wav = None
        document_wav_length = 0

        if "speech" in input_mode:
            # Process question audio
            if audio_data.get("question_audio") is not None:
                question_raw_wav = torch.tensor(audio_data["question_audio"])
                question_spectrogram = self.processor(
                    audio_data["question_audio"],
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.squeeze(0)
                question_wav_length = len(question_raw_wav)

            # Process document audio
            if audio_data.get("document_audio") is not None:
                document_raw_wav = torch.tensor(audio_data["document_audio"])
                document_spectrogram = self.processor(
                    audio_data["document_audio"],
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.squeeze(0)
                document_wav_length = len(document_raw_wav)

        # Process examples
        examples_speech = []
        if examples_audio and len(examples_audio) > 0:
            for example_audio in examples_audio:
                # Process both question and document audio for each example
                example_data = {
                    "question": {
                        "raw_wav": None,
                        "spectrogram": None,
                        "wav_length": 0
                    },
                    "document": {
                        "raw_wav": None,
                        "spectrogram": None,
                        "wav_length": 0
                    }
                }

                if example_audio.get("question_audio") is not None:
                    q_raw_wav = torch.tensor(example_audio["question_audio"])
                    example_data["question"] = {
                        "raw_wav": q_raw_wav,
                        "spectrogram": self.processor(
                            example_audio["question_audio"],
                            sampling_rate=16000,
                            return_tensors="pt"
                        ).input_features.squeeze(0),
                        "wav_length": len(q_raw_wav)
                    }

                if example_audio.get("document_audio") is not None:
                    d_raw_wav = torch.tensor(example_audio["document_audio"])
                    example_data["document"] = {
                        "raw_wav": d_raw_wav,
                        "spectrogram": self.processor(
                            example_audio["document_audio"],
                            sampling_rate=16000,
                            return_tensors="pt"
                        ).input_features.squeeze(0),
                        "wav_length": len(d_raw_wav)
                    }

                examples_speech.append(example_data)

        self.batch_counter += 1
        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "question_spectrogram": question_spectrogram,
            "question_raw_wav": question_raw_wav,
            "question_wav_length": question_wav_length,
            "document_spectrogram": document_spectrogram,
            "document_raw_wav": document_raw_wav,
            "document_wav_length": document_wav_length,
            "examples_speech": examples_speech,
            "num_examples": len(examples_speech),
            "completion": completion
        }

    def _process_default_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """Original process_inputs implementation"""
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
        
        if audio is not None and "speech" in input_mode:
            # Match _process_speech in salmon_datasets.py
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
                     input_mode: str = "speech_and_text",
                     fewshot_mode: str = "text",
                     dataset_type: Optional[DatasetType] = None,
                     **kwargs) -> str:
        if dataset_type == DatasetType.SQA:
            return self._format_sqa_prompt(template, text, examples, input_mode, fewshot_mode, **kwargs)
        else:
            # Use default for both VP-NEL and other datasets
            return self._format_default_prompt(template, text, examples, input_mode, fewshot_mode, **kwargs)

    def _format_sqa_prompt(self, template: str, text: str, examples: Optional[List[Dict]], 
                          input_mode: str, fewshot_mode: str, **kwargs):
        """Format prompt for SQA dataset"""
        question = kwargs.get('question', '')
        
        # Format examples if provided
        examples_text = ""
        if examples and len(examples) > 0:
            if fewshot_mode == "speech":
                examples_text = "\n\n".join([
                    f"Document: <Speech><Document{i}></Speech>\n"
                    f"Question: ><Speech><Question{i}></Speech>\n"
                    f"Output: {example.get('completion', '')}"
                    for i, example in enumerate(examples)
                ])
            else:
                examples_text = "\n\n".join([
                    f"Document: {example.get('document', '')}\n"
                    f"Question: {example.get('question', '')}\n"
                    f"Output: {example.get('completion', '')}"
                    for example in examples
                ])
            
            examples_text = f"\nHere are few examples to learn from:\n{examples_text}\n\n"

        # Create input section based on input mode
        if input_mode == "speech_and_text":
            input_section = (
                f"Document: <Speech><Document></Speech>\n"
                f"Document text: {text}\n"
                f"Question: <Speech><Question></Speech>\n"
                f"Question text: {question}"
                
            )
        elif input_mode == "text_only":
            # input_section = f"Question: {question}\nDocument: {text}"
            input_section = f"\nDocument: {text}\nQuestion: {question}"
        else:  # speech_only
            input_section = f"\nDocument: <Speech><Document></Speech>\n Question: <Speech><Question></Speech>"

        # Create the final prompt
        prompt = f"{template}\n{examples_text} Now analyze this input:\n{input_section}\nOutput:"
        
        return prompt

    def _format_default_prompt(self, template: str, text: str, examples: Optional[List[Dict]], 
                             input_mode: str, fewshot_mode: str, **kwargs):
        """Format prompt for default case (classification tasks)"""
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
        
        # Create input section based on input mode
        if input_mode == "speech_and_text":
            input_section = f"<Speech><SpeechHere></Speech>\nTranscript: {text}"
        elif input_mode == "text_only":
            input_section = f"Text: {text}"
        else:  # speech_only
            input_section = "<Speech><SpeechHere></Speech>"
        
        # Create the final prompt
        prompt = f"{template}\n{examples_text}Now analyze this input:\n{input_section}\nOutput:"
        
        return prompt
    
    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Route to appropriate collate function based on dataset type"""
        if batch_items[0].get("dataset_type") == DatasetType.SQA:
            return self._collate_sqa_batch(batch_items)
        else:
            # Use default for both VP-NEL and other datasets
            return self._collate_default_batch(batch_items)

    def _collate_default_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Original collate_batch implementation for default case"""
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

    def _collate_sqa_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch for SQA dataset"""
        batch = {}
        
        # Process text inputs
        batch["input_ids"] = torch.stack([item["input_ids"] for item in batch_items])
        batch["attention_mask"] = torch.stack([item["attention_mask"] for item in batch_items])
        
        # Check for valid speech data
        has_valid_speech = all(
            item.get('question_spectrogram') is not None and 
            item.get('document_spectrogram') is not None 
            for item in batch_items
        )
        
        if has_valid_speech:
            # Process question audio
            q_wav_lengths = torch.tensor([item['question_wav_length'] for item in batch_items])
            q_raw_wavs = [item['question_raw_wav'] for item in batch_items]
            q_raw_wavs_padded = pad_sequence(q_raw_wavs, batch_first=True, padding_value=0)
            q_padding_mask = torch.arange(q_raw_wavs_padded.size(1)).unsqueeze(0) >= q_wav_lengths.unsqueeze(1)
            q_spectrograms = torch.stack([item['question_spectrogram'] for item in batch_items])
            
            # Process document audio
            d_wav_lengths = torch.tensor([item['document_wav_length'] for item in batch_items])
            d_raw_wavs = [item['document_raw_wav'] for item in batch_items]
            d_raw_wavs_padded = pad_sequence(d_raw_wavs, batch_first=True, padding_value=0)
            d_padding_mask = torch.arange(d_raw_wavs_padded.size(1)).unsqueeze(0) >= d_wav_lengths.unsqueeze(1)
            d_spectrograms = torch.stack([item['document_spectrogram'] for item in batch_items])
            
            batch.update({
                "question_wav_lengths": q_wav_lengths,
                "question_raw_wav": q_raw_wavs_padded,
                "question_padding_mask": q_padding_mask,
                "question_spectrogram": q_spectrograms,
                "document_wav_lengths": d_wav_lengths,
                "document_raw_wav": d_raw_wavs_padded,
                "document_padding_mask": d_padding_mask,
                "document_spectrogram": d_spectrograms,
            })
        
        # Process examples if they exist
        max_examples = max(item['num_examples'] for item in batch_items)
        if max_examples > 0:
            has_speech_examples = any(
                'examples_speech' in item and 
                len(item['examples_speech']) > 0 and
                all(example.get('question', {}).get('spectrogram') is not None and 
                    example.get('document', {}).get('spectrogram') is not None 
                    for example in item['examples_speech'])
                for item in batch_items
            )
            
            if has_speech_examples:
                # Find max lengths for padding
                max_q_length = max(
                    example['question']['wav_length']
                    for item in batch_items if 'examples_speech' in item
                    for example in item['examples_speech'][:item['num_examples']]
                )
                max_d_length = max(
                    example['document']['wav_length']
                    for item in batch_items if 'examples_speech' in item
                    for example in item['examples_speech'][:item['num_examples']]
                )
                
                example_q_specs, example_q_wavs, example_q_masks = [], [], []
                example_d_specs, example_d_wavs, example_d_masks = [], [], []
                
                for item in batch_items:
                    examples = item.get('examples_speech', [])[:item['num_examples']]
                    batch_q_specs, batch_q_wavs, batch_q_masks = [], [], []
                    batch_d_specs, batch_d_wavs, batch_d_masks = [], [], []
                    
                    for example in examples:
                        # Process question
                        q_spec = example['question']['spectrogram']
                        q_wav = example['question']['raw_wav']
                        q_wav_length = example['question']['wav_length']
                        
                        batch_q_specs.append(q_spec)
                        if q_wav.size(0) < max_q_length:
                            q_wav = F.pad(q_wav, (0, max_q_length - q_wav.size(0)), value=0)
                        batch_q_wavs.append(q_wav)
                        q_mask = torch.arange(max_q_length, device=q_wav.device) >= q_wav_length
                        batch_q_masks.append(q_mask)
                        
                        # Process document
                        d_spec = example['document']['spectrogram']
                        d_wav = example['document']['raw_wav']
                        d_wav_length = example['document']['wav_length']
                        
                        batch_d_specs.append(d_spec)
                        if d_wav.size(0) < max_d_length:
                            d_wav = F.pad(d_wav, (0, max_d_length - d_wav.size(0)), value=0)
                        batch_d_wavs.append(d_wav)
                        d_mask = torch.arange(max_d_length, device=d_wav.device) >= d_wav_length
                        batch_d_masks.append(d_mask)
                    
                    # Pad to max_examples
                    while len(batch_q_specs) < max_examples:
                        batch_q_specs.append(torch.zeros_like(batch_q_specs[0]) if batch_q_specs else torch.zeros((80, 3000)))
                        batch_q_wavs.append(torch.zeros(max_q_length, device=q_wav.device))
                        batch_q_masks.append(torch.ones(max_q_length, device=q_wav.device, dtype=torch.bool))
                        
                        batch_d_specs.append(torch.zeros_like(batch_d_specs[0]) if batch_d_specs else torch.zeros((80, 3000)))
                        batch_d_wavs.append(torch.zeros(max_d_length, device=d_wav.device))
                        batch_d_masks.append(torch.ones(max_d_length, device=d_wav.device, dtype=torch.bool))
                    
                    example_q_specs.append(torch.stack(batch_q_specs))
                    example_q_wavs.append(torch.stack(batch_q_wavs))
                    example_q_masks.append(torch.stack(batch_q_masks))
                    
                    example_d_specs.append(torch.stack(batch_d_specs))
                    example_d_wavs.append(torch.stack(batch_d_wavs))
                    example_d_masks.append(torch.stack(batch_d_masks))
                
                batch.update({
                    "example_question_spectrograms": torch.stack(example_q_specs),
                    "example_question_wavs": torch.stack(example_q_wavs),
                    "example_question_padding_masks": torch.stack(example_q_masks),
                    "example_document_spectrograms": torch.stack(example_d_specs),
                    "example_document_wavs": torch.stack(example_d_wavs),
                    "example_document_padding_masks": torch.stack(example_d_masks),
                })

        
        # Add non-tensor data
        batch["num_examples"] = torch.tensor([item["num_examples"] for item in batch_items])
        for key in ["prompt", "completion", "text", "question","dataset_type"]:
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