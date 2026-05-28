import logging
import random
import time
from typing import Any, Dict, List

import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset

from config.data_config.master_config import DatasetSplit, DatasetType, get_dataset_config
from .model_processors import ModelProcessor

logger = logging.getLogger(__name__)


def convert_ner_to_dict(text: str, ner_data: Dict) -> Dict[str, List[str]]:
    """Convert NER span annotations into a compact tag-to-phrases dictionary."""
    result: Dict[str, List[str]] = {}
    for tag, start, length in zip(ner_data["type"], ner_data["start"], ner_data["length"]):
        phrase = text[start : start + length]
        if phrase.strip():
            result.setdefault(tag, []).append(phrase)
    return result


class BaseMultiTaskDataset(Dataset):
    """Base class for single-task formatting with optional few-shot support."""

    def __init__(
        self,
        dataset_type: DatasetType,
        dataset,
        processor: ModelProcessor,
        input_mode: str = "speech_only",
        fewshot_mode: str = "text",
        num_examples: int = 5,
        random_examples: bool = False,
        split: DatasetSplit = DatasetSplit.TEST,
        model_type: str = "salmonn",
        run_name: str = "",
        randomize_swap: bool = False,
        is_training: bool = False,
    ):
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.processor = processor
        self.input_mode = input_mode
        self.fewshot_mode = fewshot_mode
        self.num_examples = num_examples
        self.random_examples = random_examples
        self.split = split
        self.model_type = model_type.lower()
        self.run_name = run_name
        self.randomize_swap = randomize_swap
        self.is_training = is_training

        self.config = get_dataset_config(dataset_type)
        self.current_config = self.config

        self.audio_lookup = None
        self.audio_index_map = None
        if self.fewshot_mode == "speech":
            audio_lookup_path = self.config.get_audio_lookup_path(self.split)
            if audio_lookup_path:
                load_time = time.time()
                self.audio_lookup = load_from_disk(audio_lookup_path)
                self.audio_index_map = {str(idx): i for i, idx in enumerate(self.audio_lookup["index"])}
                logger.info(
                    "Initialized audio lookup index in %.3fs for %s",
                    time.time() - load_time,
                    self.dataset_type,
                )

    def __len__(self):
        return len(self.dataset)

    def _get_audio_by_index(self, index_str: str):
        if not index_str or self.audio_lookup is None or self.audio_index_map is None:
            return None
        lookup_idx = self.audio_index_map.get(index_str)
        if lookup_idx is None:
            return None
        try:
            return self.audio_lookup[lookup_idx]["audio"]
        except Exception as exc:
            logger.error("Error loading audio for index %s: %s", index_str, exc)
            return None

    def _select_examples(self, few_shot_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not few_shot_examples:
            return []
        if self.random_examples:
            count = random.randint(0, self.num_examples)
            if count == 0:
                return []
            return random.sample(few_shot_examples, min(count, len(few_shot_examples)))
        return few_shot_examples[: self.num_examples]

    def _format_label(self, example_or_label, is_example: bool = True, current_mapping=None, text: str = None):
        label = example_or_label["label"] if is_example else example_or_label

        if self.dataset_type == DatasetType.VOXPOPULI and isinstance(label, dict):
            if not is_example:
                label = convert_ner_to_dict(text or "", label)
            label_keys = [k for k, v in label.items() if v]
            label = ", ".join(label_keys) if label_keys else "none"

        if isinstance(label, list):
            label = ", ".join(label)

        label = str(label).lower()

        mapping_to_use = current_mapping if current_mapping is not None else self.config.label_mapping
        if mapping_to_use:
            if "," in label:
                parts = [part.strip().lower() for part in label.split(",")]
                label = ", ".join(mapping_to_use.get(part, part) for part in parts)
            else:
                label = mapping_to_use.get(label, label)

        return label

    def __getitem__(self, idx):
        return self._process_default_item(self.dataset[idx])

    def _process_default_item(self, item):
        current_config = self.current_config
        formatted_examples: List[Dict[str, str]] = []
        examples_audio = []

        if self.audio_lookup is not None and self.num_examples > 0:
            total_examples = len(self.audio_lookup)
            count = random.randint(0, self.num_examples) if self.random_examples else self.num_examples
            sampled_indices = random.sample(range(total_examples), min(count, total_examples)) if count > 0 else []

            for sample_idx in sampled_indices:
                example = self.audio_lookup[sample_idx]
                formatted_examples.append(
                    {
                        "text": example[current_config.text_key],
                        "label": self._format_label(
                            example[current_config.completion_key],
                            is_example=False,
                            current_mapping=current_config.label_mapping,
                            text=example[current_config.text_key],
                        ),
                    }
                )
                if self.fewshot_mode == "speech" and "audio" in example:
                    examples_audio.append(example["audio"]["array"])
        else:
            selected_examples = self._select_examples(item.get("few_shot_examples", []))
            for example in selected_examples:
                formatted_examples.append(
                    {
                        "text": example["text"],
                        "label": self._format_label(
                            example,
                            is_example=True,
                            current_mapping=current_config.label_mapping,
                        ),
                    }
                )
            examples_audio = self._get_examples_audio(selected_examples)

        prompt = self.processor.format_prompt(
            template=current_config.prompt_template,
            text=item[current_config.text_key],
            examples=formatted_examples,
            input_mode=self.input_mode,
            fewshot_mode=self.fewshot_mode,
            dataset_type=self.dataset_type,
        )

        formatted_completion = self._format_label(
            item[current_config.completion_key],
            is_example=False,
            current_mapping=current_config.label_mapping,
            text=item[current_config.text_key],
        )

        # IMPORTANT:
        # Do NOT tokenize here.
        # Tokenization must happen in processor.collate_batch so SymbolManager can
        # rewrite prompt/completion BEFORE tokenization.
        return {
            "prompt": prompt,
            "text": item[current_config.text_key],
            "completion": formatted_completion,
            "audio": self._get_main_audio(item),
            "examples_audio": examples_audio if examples_audio else None,
            "input_mode": self.input_mode,
            "fewshot_mode": self.fewshot_mode,
            "dataset_type": self.dataset_type,
            "is_training": self._is_training(),
        }

    def _get_main_audio(self, item):
        if "speech" in self.input_mode and "audio" in item:
            return item["audio"]["array"]
        return None

    def _get_examples_audio(self, selected_examples):
        if self.fewshot_mode != "speech":
            return None

        examples_audio = []
        for example in selected_examples:
            if "index" not in example:
                continue
            example_audio = self._get_audio_by_index(example["index"])
            if example_audio is None:
                continue
            array = example_audio["array"]
            examples_audio.append(np.array(array) if isinstance(array, list) else array)

        return examples_audio or None

    def _is_training(self):
        return self.is_training


class TrainingBaseDataset(BaseMultiTaskDataset):
    """Single-task dataset wrapper that marks items as training=True."""

    def _is_training(self):
        return True


class MultiTaskDataset(Dataset):
    """Dataset that combines multiple datasets with balanced or sequential sampling."""

    def __init__(
        self,
        datasets: Dict[DatasetType, BaseMultiTaskDataset],
        balance_datasets: bool = True,
        interleave: bool = True,
    ):
        self.datasets = datasets
        self.dataset_types = list(datasets.keys())
        self.balance_datasets = balance_datasets
        self.interleave = interleave

        self.dataset_sizes = {dt: len(dataset) for dt, dataset in datasets.items()}

        if self.balance_datasets:
            self.max_size = max(self.dataset_sizes.values())
            self.total_size = self.max_size * len(self.dataset_types)
            self.dataset_indices = {}
            for dt in self.dataset_types:
                size = self.dataset_sizes[dt]
                repeats = (self.max_size + size - 1) // size
                self.dataset_indices[dt] = np.tile(np.arange(size), repeats)[: self.max_size]
                np.random.shuffle(self.dataset_indices[dt])
        elif self.interleave:
            self.max_size = max(self.dataset_sizes.values())
            self.total_size = sum(self.dataset_sizes.values())
            self.dataset_indices = {}
            for dt in self.dataset_types:
                size = self.dataset_sizes[dt]
                self.dataset_indices[dt] = np.arange(size)
                np.random.shuffle(self.dataset_indices[dt])
        else:
            self.total_size = sum(self.dataset_sizes.values())
            self.index_mapping = []
            for dt in self.dataset_types:
                for local_idx in range(self.dataset_sizes[dt]):
                    self.index_mapping.append((dt, local_idx))

        logger.info("Created MultiTaskDataset with %d tasks", len(self.dataset_types))
        logger.info(
            "Sampling mode: %s, %s",
            "balanced" if balance_datasets else "unbalanced",
            "interleaved" if interleave else "sequential",
        )
        for dt in self.dataset_types:
            logger.info("  - %s: %d examples", dt, self.dataset_sizes[dt])
        logger.info("Total examples per epoch: %d", self.total_size)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if self.balance_datasets or self.interleave:
            dataset_idx = idx % len(self.dataset_types)
            dataset_type = self.dataset_types[dataset_idx]

            if self.balance_datasets:
                local_idx = idx // len(self.dataset_types)
                actual_idx = int(self.dataset_indices[dataset_type][local_idx % self.max_size])
            else:
                local_idx = idx // len(self.dataset_types)
                dataset_size = len(self.dataset_indices[dataset_type])
                actual_idx = int(self.dataset_indices[dataset_type][local_idx % dataset_size])

            item = self.datasets[dataset_type][actual_idx]
        else:
            dataset_type, local_idx = self.index_mapping[idx]
            item = self.datasets[dataset_type][int(local_idx)]

        if "dataset_type" not in item:
            item["dataset_type"] = dataset_type
        return item

    def on_epoch_end(self):
        if self.balance_datasets or self.interleave:
            for dt in self.dataset_types:
                np.random.shuffle(self.dataset_indices[dt])