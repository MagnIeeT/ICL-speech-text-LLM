import abc
import logging
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Tuple, Union
import time
import numpy as np

from .master_config import DatasetType, DatasetSplit, get_dataset_config, get_swap_config
from .model_processors import ModelProcessor
from datasets import load_from_disk

# from ..ZscriptsINTERSPEECH.utils.generate_fewshots import convert_ner_to_dict

logger = logging.getLogger(__name__)


def convert_ner_to_dict(text: str, ner_data: Dict) -> Dict[str, List[str]]:
    """
    Convert NER data from start/length format to {tag: [phrases]} format.
    Only includes tags that have actual phrases (not None).
    
    Args:
        text: The input text
        ner_data: Dictionary containing tag-phrase mappings
    
    Returns:
        Dictionary mapping entity types to lists of phrases, excluding empty tags
    """
    result = {}
    
    # For the original start/length format
    for tag, start, length in zip(ner_data['type'], ner_data['start'], ner_data['length']):
        # Extract the phrase using start and length
        phrase = text[start:start + length]
        
        # Only add non-empty phrases
        if phrase.strip():
            if tag not in result:
                result[tag] = []
            result[tag].append(phrase)
    
    return result


class BaseMultiTaskDataset(Dataset):
    """Base class for multi-task datasets"""
    
    def __init__(
        self, 
        dataset_type: DatasetType, 
        dataset, 
        processor,
        input_mode: str = 'speech_only',
        fewshot_mode: str = 'text', 
        num_examples: int = 5, 
        random_examples: bool = False, 
        split: DatasetSplit = DatasetSplit.TEST, 
        model_type: str = "salmonn", 
        run_name: str = "",
        randomize_swap: bool = False
    ):
        """
        Initialize the base multi-task dataset.
        
        Args:
            dataset_type: Type of dataset (e.g., VOXCELEB, HVB)
            dataset: The actual dataset object
            processor: Model-specific processor for text/audio
            input_mode: 'speech_and_text', 'speech_only', or 'text_only'
            fewshot_mode: 'text' or 'speech' for few-shot examples
            num_examples: Number of few-shot examples to use
            random_examples: Whether to randomly select few-shot examples
            split: Dataset split (TRAIN, VAL, TEST)
            model_type: Type of model ('salmonn', 'qwen2', etc.)
            run_name: Name of the run (for logging)
            randomize_swap: Whether to randomize the swap configuration
        """
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.processor = processor
        self.input_mode = input_mode
        self.fewshot_mode = fewshot_mode
        self.num_examples = num_examples
        # self.random_examples = random_examples
        self.random_examples = False
        self.split = split
        self.model_type = model_type.lower()
        self.run_name = run_name
        # self.randomize_swap = randomize_swap
        self.randomize_swap = False
        
        # Get base dataset configuration
        self.config = get_dataset_config(dataset_type)
        
        # For non-swap datasets, we can get the config once
        self.is_swap_dataset = dataset_type in [DatasetType.VOXCELEB_SWAP, DatasetType.HVB_SWAP, DatasetType.VOXPOPULI_SWAP, DatasetType.MELD_EMOTION_SWAP]
        if not self.is_swap_dataset:
            self.current_config = self.config
        
        # Pre-load audio lookup if using speech examples
        self.audio_lookup = None
        self.audio_index_map = None
        if fewshot_mode == 'speech':
            audio_lookup_path = self.config.get_audio_lookup_path(self.split)
        else:
            audio_lookup_path = None
        if audio_lookup_path:
            load_time = time.time()
            if  self.dataset_type in [
                DatasetType.SQA, 
                # DatasetType.VOXPOPULI_NEL,
                # DatasetType.MELD,
                # DatasetType.HVB,
                # DatasetType.MELD_EMOTION,
                # DatasetType.MELD_GREEK,
                # DatasetType.MELD_EMOTION_GREEK,
                # DatasetType.VOXPOPULI,
                # DatasetType.VOXPOPULI_GREEK,
                # DatasetType.VOXPOPULI_SWAP,
                # DatasetType.MELD_EMOTION_SWAP
            ]:
                # For SQA and VP-NEL, just load the dataset
                self.audio_lookup = load_from_disk(audio_lookup_path)
                logger.info(f"self.audio_lookup keys: {self.audio_lookup[0].keys()}")
                logger.info(f"Initialized audio lookup dataset for {self.dataset_type} in {time.time() - load_time:.3f}s")
            else:
                # For other datasets, create index map
                self.audio_lookup = load_from_disk(audio_lookup_path)
                self.audio_index_map = {str(idx): i for i, idx in enumerate(self.audio_lookup['index'])}
                logger.info(f"Initialized audio lookup dataset with index map in {time.time() - load_time:.3f}s")

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.dataset)

    def _get_audio_by_index(self, index_str):
        start_time = time.time()
        
        if not index_str:
            return None
        
        if self.audio_lookup is None:
            logger.warning(f"Audio lookup not initialized for {self.dataset_type}")
            return None
        
        try:
            # lookup_time = time.time()
            lookup_idx = self.audio_index_map.get(index_str)
            if lookup_idx is None:
                logger.warning(f"No matching audio found for index {index_str}")
                return None
            
            audio = self.audio_lookup[lookup_idx]['audio']
            # logger.info(f"Audio lookup took: {time.time() - lookup_time:.3f}s")
            return audio
        
        except Exception as e:
            logger.error(f"Error loading audio for index {index_str}: {e}")
            return None

    def _select_examples(self, few_shot_examples):
        """Helper method to select a random number of examples between 0 and num_examples"""
        if self.random_examples:
            # First select a random number between 0 and num_examples
            random_count = random.randint(0, self.num_examples)
            if random_count > 0:
                # If random count is greater than 0, select that many examples (or all available)
                num_to_select = min(random_count, len(few_shot_examples))
                return random.sample(few_shot_examples, num_to_select) if num_to_select > 0 else []
            else:
                # If random count is 0, return empty list
                return []
        # Original behavior for non-random mode
        return few_shot_examples[:self.num_examples]

    def _format_label(self, example_or_label, is_example=True, current_mapping=None, text=None):
        """Format label based on dataset type and configuration."""
        # Extract the label from example if needed
        if is_example:
            label = example_or_label['label']
        else:
            label = example_or_label

        # Handle special output formats
        if hasattr(self.current_config, 'output_format'):
            # SQA timestamps format: "start_time end_time"
            if self.current_config.output_format == 'timestamps_pair':
                return f"{label}"

            # VP-NEL entity timestamps format: "TYPE: start end; TYPE2: start2 end2"
            elif self.current_config.output_format == 'entity_timestamps':
                if not label:  # Empty list
                    return 'none'
                formatted_spans = []
                for span in label:
                    formatted_spans.append(f"{span['label']}: {span['time_span'][0]} {span['time_span'][1]}")
                return '; '.join(formatted_spans)

        # Handle different dataset types first
        if self.dataset_type in [DatasetType.VOXPOPULI, DatasetType.VOXPOPULI_SWAP, DatasetType.VOXPOPULI_GREEK]:
            if isinstance(label, dict):
                # Filter out keys with None values
                if not is_example:
                    label = convert_ner_to_dict(text, label)
                label_dict = [k for k, v in label.items() if v]
                label= ', '.join(label_dict) if label_dict else 'none'
        
        # Handle list-type labels (for HVB datasets)
        if isinstance(label, list):
            # Convert list to comma-separated string
            label = ', '.join(label)
        
        # Normalize case to lowercase
        label = label.lower()
        
        # Apply label mapping if needed
        mapping_to_use = current_mapping if current_mapping is not None else self.config.label_mapping
        if mapping_to_use and isinstance(label, str):
            if "," in label:
                # Handle multi-label case (e.g., HVB)
                parts = [part.strip().lower() for part in label.split(",")]
                mapped_parts = [mapping_to_use.get(part, part) for part in parts]
                label = ", ".join(mapped_parts)
            else:
                # Single label case
                label = mapping_to_use.get(label.lower(), label.lower())
                
        return label

    def __getitem__(self, idx):
        if self.is_swap_dataset:
            self.current_config = get_swap_config(self.dataset_type, self.randomize_swap)

        # Get item from dataset
        item = self.dataset[idx]
        
        # Handle different dataset types
        if self.dataset_type == DatasetType.SQA:
            return self._process_sqa_item(item, idx)
        else:
            # Use default for both VP-NEL and other datasets
            return self._process_default_item(item, idx)

    def _process_sqa_item(self, item, idx):
        """Special processing for SQA dataset"""
        current_config = self.current_config
        
        # Instead of getting few_shot_examples from item, randomly sample from audio lookup
        if self.audio_lookup is not None and self.num_examples > 0:
            # Randomly sample indices from audio lookup
            total_examples = len(self.audio_lookup)
            if self.random_examples:
                # First select a random number between 0 and num_examples
                random_count = random.randint(0, self.num_examples)
                if random_count > 0:
                    # If random count is greater than 0, select that many examples
                    num_to_select = min(random_count, total_examples)
                    sampled_indices = random.sample(range(total_examples), num_to_select)
                else:
                    # If random count is 0, select no examples
                    sampled_indices = []
            else:
                # Original behavior - select exactly num_examples or all available
                sampled_indices = random.sample(range(total_examples), min(self.num_examples, total_examples))
            
            # Create few-shot examples from sampled indices
            formatted_examples = []
            examples_audio = []
            
            for sample_idx in sampled_indices:
                example = self.audio_lookup[sample_idx]
                formatted_example = {
                    "question": example[current_config.additional_text_keys['question']],
                    "document": example[current_config.text_key],
                    "completion": self._format_label(example[current_config.completion_key],
                        is_example=False, 
                        current_mapping=current_config.label_mapping
                    )
                }
                formatted_examples.append(formatted_example)
                
                if self.fewshot_mode == 'speech':
                    example_audio = {
                        'question_audio': example['question_audio']['array'] if 'question_audio' in example else None,
                        'document_audio': example['document_audio']['array'] if 'document_audio' in example else None
                    }
                    examples_audio.append(example_audio)
        else:
            formatted_examples = []
            examples_audio = None
        
        # Create prompt using both question and document
        prompt = self.processor.format_prompt(
            template=current_config.prompt_template,
            text=item[current_config.text_key],  # document text
            question=item[current_config.additional_text_keys['question']],  # question text
            examples=formatted_examples,
            input_mode=self.input_mode,
            fewshot_mode=self.fewshot_mode,
            dataset_type=DatasetType.SQA
        )
        
        # Get audio data for both question and document
        audio_data = {
            'question_audio': self._get_audio_by_key(item, 'question_audio'),
            'document_audio': self._get_audio_by_key(item, 'document_audio')
        }
        
        # Process inputs
        inputs = self.processor.process_inputs(
            data={
                "prompt": prompt,
                "fewshot_mode": self.fewshot_mode,
                "input_mode": self.input_mode,
                "completion": self._format_label(item[current_config.completion_key],
                        is_example=False, 
                        current_mapping=current_config.label_mapping
                    ),
                "audio": audio_data,
                "examples_audio": examples_audio,
                "dataset_type": self.dataset_type
            },
            is_training=self._is_training()
        )
        
        # Build result with additional metadata
        result = {
            "prompt": prompt,
            "text": item[current_config.text_key],
            "question": item[current_config.additional_text_keys['question']],
            "completion": item[current_config.completion_key],
            "dataset_type": self.dataset_type,
            "unique_id": item[current_config.additional_metadata_keys['unique_id']],
            **inputs
        }
        
        return result


    def _get_audio_by_key(self, item, key):
        """Helper to get audio by key from item"""
        if key in item and item[key] is not None:
            return item[key]['array']
        return None

    def _process_default_item(self, item, idx):
        """Processing logic for VP-NEL and other datasets"""
        current_config = self.current_config
        
        # Get examples either from audio_lookup or few_shot_examples
        formatted_examples = []
        examples_audio = []
        
        if (
            # self.dataset_type == DatasetType.VOXPOPULI_NEL or 
        # self.dataset_type == DatasetType.MELD or 
        
        # self.DatasetType.HVB or
        # self.dataset_type == DatasetType.MELD_EMOTION or
        # self.dataset_type == DatasetType.MELD_GREEK 
        # self.dataset_type == DatasetType.MELD_EMOTION_GREEK or 
        # self.dataset_type == DatasetType.VOXPOPULI or
        # self.dataset_type == DatasetType.VOXPOPULI_GREEK or
        # self.dataset_type == DatasetType.VOXPOPULI_SWAP or
        # self.dataset_type == DatasetType.MELD_EMOTION_SWAP
        self.dataset_type == DatasetType.SQA 
        ) and self.audio_lookup is not None and self.num_examples > 0:
            
            # Random sampling from audio_lookup
            total_examples = len(self.audio_lookup)
            if self.random_examples:
                # First select a random number between 0 and num_examples
                random_count = random.randint(0, self.num_examples)
                if random_count > 0:
                    # If random count is greater than 0, select that many examples
                    num_to_select = min(random_count, total_examples)
                    sampled_indices = random.sample(range(total_examples), num_to_select)
                else:
                    # If random count is 0, select no examples
                    sampled_indices = []
            else:
                # Original behavior - select exactly num_examples or all available
                sampled_indices = random.sample(range(total_examples), min(self.num_examples, total_examples))
            
            for sample_idx in sampled_indices:
                example = self.audio_lookup[sample_idx]
                # if self.dataset_type == DatasetType.VOXPOPULI_SWAP:
                    # logger.info(f"Sample {sample_idx} example: {example.keys()}")
                formatted_example = {
                    "text": example[current_config.text_key],
                    "label": self._format_label( example[current_config.completion_key],
                        is_example=False,
                        current_mapping=current_config.label_mapping,
                        text=example[current_config.text_key]
                    )
                }
                formatted_examples.append(formatted_example)
                
                if self.fewshot_mode == 'speech':
                    example_audio = example['audio']['array'] if 'audio' in example else None
                    if example_audio is not None:
                        examples_audio.append(example_audio)
        else:
            # Default: Use few_shot_examples from item
            few_shot_examples = item.get('few_shot_examples', [])
            selected_examples = self._select_examples(few_shot_examples)
            
            for example in selected_examples:
                formatted_example = {
                    "text": example['text'],
                    "label": self._format_label(example, is_example=True, current_mapping=current_config.label_mapping)
                }
                formatted_examples.append(formatted_example)
            
            # Get examples audio using existing method
            examples_audio = self._get_examples_audio(selected_examples)
        
        # Create prompt
        prompt = self.processor.format_prompt(
            template=current_config.prompt_template,
            text=item[current_config.text_key],
            examples=formatted_examples,
            input_mode=self.input_mode,
            fewshot_mode=self.fewshot_mode,
            dataset_type=self.dataset_type
        )
        
        # Get audio data
        audio_data = self._get_main_audio(item)
        
        # Format the completion/label
        formatted_completion = self._format_label(
            item[current_config.completion_key], 
            is_example=False, 
            current_mapping=current_config.label_mapping,
            text=item[current_config.text_key]  # Pass text for NER cases
        )
        
        # Process inputs
        inputs = self.processor.process_inputs(
            data={
                "prompt": prompt,
                "fewshot_mode": self.fewshot_mode,
                "input_mode": self.input_mode,
                "completion": formatted_completion,  # Use formatted completion
                "audio": audio_data,
                "examples_audio": examples_audio if examples_audio else None,
                "dataset_type": self.dataset_type
            },
            is_training=self._is_training()
        )
        
        # Build result with basic fields
        result = {
            "prompt": prompt,
            "text": item[current_config.text_key],
            "completion": formatted_completion,  # Use formatted completion
            "dataset_type": self.dataset_type,
            **inputs
        }
        
        return result

    def _get_main_audio(self, item):
        """Get main audio data if needed (can be overridden)"""
        audio_data = None
        if "speech" in self.input_mode and "audio" in item:
            audio_data = item["audio"]["array"]
        return audio_data
        
    def _get_examples_audio(self, selected_examples):
        """Get audio data for examples if needed (can be overridden)"""
        examples_audio = None
        if self.fewshot_mode == 'speech':
            examples_audio = []
            # logger.info(f"Getting audio for {len(selected_examples)} examples")

            for i, example in enumerate(selected_examples):
                if 'index' in example:
                    example_audio = self._get_audio_by_index(example['index'])
                    if example_audio is not None:
                        if isinstance(example_audio['array'], list):
                            examples_audio.append(np.array(example_audio['array']))
                        else:
                            examples_audio.append(example_audio['array'])

            if not examples_audio:
                examples_audio = None

        return examples_audio
        
    def _is_training(self):
        """Whether this dataset is for training (can be overridden)"""
        return False

    def set_task(self, task_type: DatasetType):
        """
        Switch the current task/dataset type.
        
        Args:
            task_type: The dataset type to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if task_type not in self.few_shot_examples:
            logger.warning(f"Task {task_type} not found in available tasks")
            return False
        
        # Update the current dataset type
        self.dataset_type = task_type
        
        # Update the config and label mapping for the new task
        self.config = get_dataset_config(task_type)
        self.label_mapping = self.config.label_mapping
        
        logger.info(f"Switched to task: {task_type}")
        return True

    def get_available_tasks(self):
        """
        Get a list of available tasks/dataset types.
        
        Returns:
            List of available dataset types
        """
        return list(self.few_shot_examples.keys())

class MultiTaskDataset(Dataset):
    """Dataset that combines multiple datasets into one with balanced sampling"""
    
    def __init__(
        self,
        datasets: Dict[DatasetType, BaseMultiTaskDataset],
        processor,
        balance_datasets: bool = True,
        interleave: bool = True  # Add new parameter
    ):
        self.datasets = datasets
        self.dataset_types = list(datasets.keys())
        self.processor = processor
        self.balance_datasets = balance_datasets
        self.interleave = interleave
        
        # Calculate dataset sizes
        self.dataset_sizes = {dt: len(dataset) for dt, dataset in datasets.items()}
        
        if self.balance_datasets:
            # Use the maximum dataset size for balanced sampling
            self.max_size = max(self.dataset_sizes.values())
            self.total_size = self.max_size * len(self.dataset_types)
            
            # Create sampling indices for each dataset
            self.dataset_indices = {}
            for dt in self.dataset_types:
                size = self.dataset_sizes[dt]
                repeats = (self.max_size + size - 1) // size  # Ceiling division
                self.dataset_indices[dt] = np.tile(np.arange(size), repeats)[:self.max_size]
                np.random.shuffle(self.dataset_indices[dt])
        else:
            if self.interleave:
                # For interleaved but unbalanced sampling
                self.max_size = max(self.dataset_sizes.values())
                self.total_size = sum(self.dataset_sizes.values())
                
                # Create sampling indices for each dataset
                self.dataset_indices = {}
                for dt in self.dataset_types:
                    size = self.dataset_sizes[dt]
                    self.dataset_indices[dt] = np.arange(size)
                    np.random.shuffle(self.dataset_indices[dt])
            else:
                # Original sequential sampling
                self.total_size = sum(self.dataset_sizes.values())
                self.index_mapping = []
                for dt in self.dataset_types:
                    for local_idx in range(self.dataset_sizes[dt]):
                        self.index_mapping.append((dt, local_idx))
        
        logger.info(f"Created MultiTaskDataset with {len(self.dataset_types)} tasks")
        logger.info(f"Sampling mode: {'balanced' if balance_datasets else 'unbalanced'}, {'interleaved' if interleave else 'sequential'}")
        for dt in self.dataset_types:
            logger.info(f"  - {dt}: {self.dataset_sizes[dt]} examples")
        logger.info(f"Total examples per epoch: {self.total_size}")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if self.balance_datasets or self.interleave:
            # Interleaved sampling (both balanced and unbalanced)
            dataset_idx = idx % len(self.dataset_types)
            dataset_type = self.dataset_types[dataset_idx]
            
            if self.balance_datasets:
                # For balanced sampling
                local_idx = idx // len(self.dataset_types)
                actual_idx = int(self.dataset_indices[dataset_type][local_idx % self.max_size])
            else:
                # For unbalanced but interleaved sampling
                local_idx = idx // len(self.dataset_types)
                dataset_size = len(self.dataset_indices[dataset_type])
                actual_idx = int(self.dataset_indices[dataset_type][local_idx % dataset_size])
            
            item = self.datasets[dataset_type][actual_idx]
        else:
            # Sequential sampling
            dataset_type, local_idx = self.index_mapping[idx]
            item = self.datasets[dataset_type][int(local_idx)]
        
        # Add the dataset type to the result if not already present
        if "dataset_type" not in item:
            item["dataset_type"] = dataset_type
        
        return item

    def on_epoch_end(self):
        """Call this at the end of each epoch to reshuffle indices"""
        if self.balance_datasets or self.interleave:
            for dt in self.dataset_types:
                np.random.shuffle(self.dataset_indices[dt])

class MultiTaskTrainingDataset(MultiTaskDataset):
    """Multi-task dataset for training"""
    
    def __init__(self, datasets: Dict[DatasetType, BaseMultiTaskDataset], processor, balance_datasets: bool = True, interleave:bool = True):
        super().__init__(datasets, processor, balance_datasets=balance_datasets, interleave=interleave)
    
    def _is_training(self):
        return True

class MultiTaskInferenceDataset(MultiTaskDataset):
    """Multi-task dataset for inference"""
    
    def __init__(self, datasets: Dict[DatasetType, BaseMultiTaskDataset], processor, balance_datasets: bool = False, interleave:bool = False):
        super().__init__(datasets, processor, balance_datasets=balance_datasets,interleave=interleave)
    
    def _is_training(self):
        return False