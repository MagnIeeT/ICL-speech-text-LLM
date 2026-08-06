import logging
import random
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset

from config.data_config.master_config import DatasetSplit, DatasetType, get_dataset_config
from .model_processors import ModelProcessor

logger = logging.getLogger(__name__)


def _apply_symbol_mapping_to_text(text: str, mapping: Dict[str, str]) -> str:
    """Replace label names with symbols in a text string (word-boundary safe)."""
    if not mapping or not isinstance(text, str):
        return text
    # Protect description text in bullet lines: "- label: DESCRIPTION" — only
    # the label token before the colon should be replaced, not the description.
    desc_store: Dict[str, str] = {}
    def _protect_desc(m: re.Match) -> str:
        key = f"__DESCPROTECT_{len(desc_store):04d}__"
        desc_store[key] = m.group(2)
        return m.group(1) + key
    protected = re.sub(r'(?m)^([ \t]*-[^:\n]+: )(.+)$', _protect_desc, text)

    placeholders: Dict[str, str] = {}
    transformed = protected
    for idx, src in enumerate(sorted(mapping.keys(), key=len, reverse=True)):
        ph = f"__SWAP_PLACEHOLDER_{idx}__"
        placeholders[ph] = mapping[src]
        transformed = re.compile(r'\b' + re.escape(src) + r'\b', re.IGNORECASE).sub(ph, transformed)
    for ph, dst in placeholders.items():
        transformed = transformed.replace(ph, dst)
    for key, val in desc_store.items():
        transformed = transformed.replace(key, val)
    return transformed


def _apply_symbol_mapping_to_prompt(prompt_obj: Any, mapping: Dict[str, str]) -> Any:
    """Walk a prompt conversation structure and apply symbol mapping to all text parts."""
    if not mapping:
        return prompt_obj
    if isinstance(prompt_obj, str):
        return _apply_symbol_mapping_to_text(prompt_obj, mapping)
    if isinstance(prompt_obj, dict) and "conversation" in prompt_obj:
        new_obj = dict(prompt_obj)
        new_conv = []
        for msg in prompt_obj.get("conversation", []):
            new_msg = dict(msg)
            if isinstance(new_msg.get("content"), str):
                new_msg["content"] = _apply_symbol_mapping_to_text(new_msg["content"], mapping)
            elif isinstance(new_msg.get("content"), list):
                new_msg["content"] = [
                    {**p, "text": _apply_symbol_mapping_to_text(p["text"], mapping)}
                    if p.get("type") == "text" else p
                    for p in new_msg["content"]
                ]
            new_conv.append(new_msg)
        new_obj["conversation"] = new_conv
        return new_obj
    return prompt_obj


def _strip_legend(template: str) -> str:
    """Remove '- label: DESCRIPTION' → '- label' from prompt bullet lines (Wei-style
    no-legend: the class meaning is no longer stated, only the label/symbol remains).
    Non-bullet lines and bullets without a colon (e.g. guideline lines) are untouched."""
    return re.sub(r'(?m)^([ \t]*-[ \t]*[^:\n]+?)[ \t]*:.*$', r'\1', template)


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
        num_examples: int = 0,
        num_examples_min: int = -1,
        random_examples: bool = False,
        split: DatasetSplit = DatasetSplit.TEST,
        model_type: str = "salmonn",
        run_name: str = "",
        randomize_swap: bool = False,
        is_training: bool = False,
        no_legend: bool = False,
        fewshot_per_class: bool = False,
    ):
        self.no_legend = no_legend
        self.fewshot_per_class = fewshot_per_class
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.processor = processor
        self.input_mode = input_mode
        self.fewshot_mode = fewshot_mode
        self.num_examples = num_examples
        self.num_examples_min = num_examples_min
        self.random_examples = random_examples
        self.split = split
        self.model_type = model_type.lower()
        self.run_name = run_name
        self.randomize_swap = randomize_swap
        self.is_training = is_training

        self.config = get_dataset_config(dataset_type)
        self.current_config = self.config

        # Symbol map set by SymbolTrainingOrchestrator at epoch start.
        # Workers inherit these at fork time — no IPC needed.
        self._no_symbols: bool = True
        self._per_instance: bool = False
        self._symbol_maps: Dict[str, Dict[str, str]] = {}   # {ds_name: {label: symbol}}
        self._symbol_pool: Dict[str, List[Dict[str, str]]] = {}  # {ds_name: [map, ...]}

        # ── Few-shot exemplar source: at runtime, sample class-balanced exemplars from
        # the TRAIN split (no separate pool → no audio duplication). Training reuses the
        # main dataset (it IS the train split); eval loads the train split from config.
        self._exemplar_ds = None
        self._exemplar_by_class: Dict[str, List[int]] = {}
        if self.num_examples > 0:
            self._init_exemplar_source()

    @staticmethod
    def _base_labels(lab):
        """Base class label(s) for single- or multi-label targets (hvb is a list)."""
        if isinstance(lab, (list, tuple)):
            return [str(x) for x in lab]
        s = str(lab)
        return [p.strip() for p in s.split(",")] if ", " in s else [s]

    def _init_exemplar_source(self):
        """Load the TRAIN split as exemplar source and index rows by base class."""
        try:
            self._exemplar_ds = self.dataset if self.is_training else load_from_disk(
                self.config.get_path(DatasetSplit.TRAIN))
        except Exception as e:  # e.g. dataset has no TRAIN split (eval-only)
            logger.warning("Few-shot: no exemplar source for %s (%s) — few-shot disabled for it",
                           self.dataset_type, e)
            self._exemplar_ds = None
            return
        by_class: Dict[str, List[int]] = {}
        for i, lab in enumerate(self._exemplar_ds[self.config.completion_key]):
            for base in self._base_labels(lab):
                by_class.setdefault(base, []).append(i)
        self._exemplar_by_class = by_class
        # AF3 accepts exactly ONE audio per prompt (processor 1:1 constraint), so audio
        # exemplars are impossible on flamingo → fall back to TEXT exemplars.
        self._audio_fewshot_ok = "audio" in self._exemplar_ds.column_names
        if self.fewshot_mode == "speech" and self.model_type == "flamingo":
            logger.warning("AF3/flamingo supports only ONE audio per prompt → audio exemplars "
                           "are not possible; few-shot for %s uses TEXT exemplars.", self.dataset_type)
            self._audio_fewshot_ok = False
        logger.info("Few-shot exemplar source for %s: %d items, %d base classes (%s)",
                    self.dataset_type, len(self._exemplar_ds), len(by_class),
                    "main/train" if self.is_training else "train-split")

    def _resolve_fewshot_k(self):
        """Number of exemplars (total, or per-class if fewshot_per_class). Random count during training."""
        k = self.num_examples
        if getattr(self, "is_training", False) and self.num_examples_min >= 0 and self.num_examples > self.num_examples_min:
            k = random.randint(self.num_examples_min, self.num_examples)
        return k

    def _sample_exemplar_rows(self, k, per_class):
        """Class-balanced random row-indices from the exemplar source (deduped for multi-label)."""
        by_class = self._exemplar_by_class
        classes = list(by_class.keys())
        random.shuffle(classes)
        chosen, seen = [], set()
        if per_class:
            for c in classes:
                for p in random.sample(by_class[c], min(k, len(by_class[c]))):
                    if p not in seen:
                        seen.add(p); chosen.append(p)
        else:
            pools = {c: random.sample(by_class[c], len(by_class[c])) for c in classes}
            ptr = {c: 0 for c in classes}
            while len(chosen) < k:
                progressed = False
                for c in classes:
                    if len(chosen) >= k:
                        break
                    while ptr[c] < len(pools[c]) and pools[c][ptr[c]] in seen:
                        ptr[c] += 1
                    if ptr[c] < len(pools[c]):
                        idx = pools[c][ptr[c]]; ptr[c] += 1
                        seen.add(idx); chosen.append(idx); progressed = True
                if not progressed:
                    break
        random.shuffle(chosen)
        return chosen

    def set_epoch_symbol_maps(
        self,
        maps: Dict[str, Dict[str, str]],
        no_symbols: bool = True,
        per_instance: bool = False,
        pool: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> None:
        """Called by the training orchestrator at the start of each epoch.
        Workers fork after this, so they see the updated maps automatically.
        """
        self._no_symbols = no_symbols
        self._per_instance = per_instance
        self._symbol_maps = maps or {}
        self._symbol_pool = pool or {}

    def _get_symbol_map_for_sample(self) -> Dict[str, str]:
        if self._no_symbols:
            return {}
        ds_name = self.dataset_type.value
        if self._per_instance:
            pool = self._symbol_pool.get(ds_name, [])
            return random.choice(pool) if pool else {}
        return self._symbol_maps.get(ds_name, {})

    def __len__(self):
        return len(self.dataset)

    def _select_examples(self, pool: List) -> List:
        if not pool:
            return []
        # Wei-style random count per prompt (training only): k ~ U[min, num_examples].
        # Eval keeps a fixed count (num_examples) so the protocol is constant.
        k = self.num_examples
        if getattr(self, "is_training", False) and self.num_examples_min >= 0 and self.num_examples > self.num_examples_min:
            k = random.randint(self.num_examples_min, self.num_examples)   # min=0 → some prompts zero-shot
        k = min(k, len(pool))
        if self.random_examples:
            return random.sample(pool, k)
        return pool[:k]

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
        
        effective_fewshot_mode = self.fewshot_mode

        if self.num_examples > 0 and self._exemplar_ds is not None and self._exemplar_by_class:
            # Runtime class-balanced random exemplars from the TRAIN split (fresh per prompt).
            rows = self._sample_exemplar_rows(self._resolve_fewshot_k(),
                                              getattr(self, "fewshot_per_class", False))
            has_audio = (self.fewshot_mode == "speech") and getattr(self, "_audio_fewshot_ok", False)
            effective_fewshot_mode = "speech" if has_audio else "text"
            for r in rows:
                ex = self._exemplar_ds[int(r)]
                fex = {
                    "text": ex[current_config.text_key],
                    "label": self._format_label(
                        ex[current_config.completion_key],
                        is_example=False,
                        current_mapping=current_config.label_mapping,
                        text=ex[current_config.text_key],
                    ),
                }
                if has_audio:
                    arr = ex["audio"]["array"]
                    fex["audio"] = arr                # flamingo format_prompt embeds example["audio"]
                    examples_audio.append(arr)        # other processors read the examples_audio list
                formatted_examples.append(fex)

        template = current_config.prompt_template
        if getattr(self, "no_legend", False):
            template = _strip_legend(template)
        # QA: the reading passage is TEXT (embedded into the system prompt via {text}); the
        # spoken question flows through the normal speech-audio path. Classification templates
        # have no {text} and are unaffected.
        if getattr(current_config, "task_type", "classification") == "qa" and "{text}" in template:
            template = template.replace("{text}", str(item[current_config.text_key]))
        prompt = self.processor.format_prompt(
            template=template,
            text=item[current_config.text_key],
            examples=formatted_examples,
            input_mode=self.input_mode,
            fewshot_mode=effective_fewshot_mode,
            dataset_type=self.dataset_type,
        )

        formatted_completion = self._format_label(
            item[current_config.completion_key],
            is_example=False,
            current_mapping=current_config.label_mapping,
            text=item[current_config.text_key],
        )

        sym_map = self._get_symbol_map_for_sample()
        if sym_map:
            prompt = _apply_symbol_mapping_to_prompt(prompt, sym_map)
            formatted_completion = _apply_symbol_mapping_to_text(formatted_completion, sym_map)

        inputs = self.processor.process_inputs(
            data={
                "prompt": prompt,
                "completion": formatted_completion,
                "audio": self._get_main_audio(item),
                "examples_audio": examples_audio if examples_audio else None,
            },
            is_training=self._is_training(),
        )

        return {
            "prompt": prompt,
            "text": item[current_config.text_key],
            "completion": formatted_completion,
            "input_mode": self.input_mode,
            "fewshot_mode": self.fewshot_mode,
            "dataset_type": self.dataset_type,
            "is_training": self._is_training(),
            **inputs, 
        }
    def _get_main_audio(self, item):
        if "speech" in self.input_mode and "audio" in item:
            return item["audio"]["array"]
        return None

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

    def set_epoch_symbol_maps(
        self,
        maps: Dict[str, Dict[str, str]],
        no_symbols: bool = True,
        per_instance: bool = False,
        pool: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> None:
        """Propagate symbol maps to all sub-datasets before epoch workers fork."""
        for sub_ds in self.datasets.values():
            sub_ds.set_epoch_symbol_maps(maps, no_symbols, per_instance, pool)

    def on_epoch_end(self):
        if self.balance_datasets or self.interleave:
            for dt in self.dataset_types:
                np.random.shuffle(self.dataset_indices[dt])