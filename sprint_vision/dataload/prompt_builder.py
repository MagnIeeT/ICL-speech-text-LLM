"""
LLaVAPromptBuilder — mirrors ICI's SalmonProcessor.format_prompt().

In ICI, SalmonProcessor.format_prompt() assembles the full prompt string
with optional few-shot example blocks before the test query:

    [ICI]
    {template}
    Here are few examples to learn from:
    Text: {example_text}
    Output: {example_label}
    ...
    Now analyze this input:
    <Speech><SpeechHere></Speech>
    Output:

This class does the equivalent for LLaVA image inputs.  Each example
requires its own <image> token so LLaVA can associate it with the
correct image tensor.  Symbol mapping (if active) is applied to the
instruction text and example labels using a two-pass replacement
that prevents cascade corruption — identical to ICI's
SymbolManager._apply_mapping_safe().

Design mirrors:
    ICI/dataload/salmon_processor.py    :: SalmonProcessor.format_prompt()
    ICI/dataload/multi_task_dataset.py  :: _process_default_item()
    ICI/models/symbolAdapter/symbol_manager.py :: _apply_mapping_safe()
"""

import re
from typing import Dict, List, Optional, Tuple

from llava.constants import DEFAULT_IMAGE_TOKEN


class LLaVAPromptBuilder:
    """
    Builds the human-turn prompt for LLaVA (training ICL + inference ICL).

    ICI-style structure — instruction/definitions appear ONCE, then compact
    example pairs, then the query (mirrors SalmonProcessor.format_prompt):

        {instruction with definitions (and symbols if active)}

        Here are few examples to learn from:

        <image>
        Answer: {example_label_0}

        <image>
        Answer: {example_label_1}

        Now analyze this image:

        <image>

    For zero-shot (no examples) only `{instruction}` + the query `<image>` are
    emitted. The instruction is NOT repeated per shot — that kept the long
    definition block from overflowing model_max_length.
    image_paths is ordered [example_0, ..., example_k, test_image], matching the
    <image> token order so LLaVA pairs each image with the right tensor.
    """

    def build(
        self,
        instruction: str,
        examples: List[Dict],
        test_image_path: str,
        symbol_mappings: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Build the full human-turn prompt and the ordered image path list.

        Args:
            instruction:     Instruction text extracted from the JSON
                             (e.g. "Is there a tumor? Answer with the label
                             only (0 for No, 1 for Yes).").
                             Symbol replacement is applied here when
                             symbol_mappings is provided.
            examples:        Dicts with "image" and "label" keys, as returned
                             by ExampleSelector.select().  Empty list → 0-shot.
            test_image_path: Relative path of the test image
                             (same root as image_folder in sprint_eval.py).
            symbol_mappings: Optional {"0": "kxzy", "1": "wxab"}.
                             Applied to instruction text AND example labels.
                             None → regular strategy, no substitution.

        Returns:
            (prompt_str, image_paths)
            prompt_str  — complete human-turn string with <image> tokens.
            image_paths — [example_img_0, ..., test_img] in the same order
                          as the <image> tokens in prompt_str.
                          Pass this directly to process_images().
        """
        # ICI-style structure: the instruction (with the definition block) appears
        # ONCE as a preamble, then the K example pairs (image + label), then the
        # query image — mirroring SalmonProcessor.format_prompt:
        #   {template}
        #   Here are few examples to learn from:
        #   <Speech><Example_i></Speech>\nOutput: {label_i}   (× K)
        #   Now analyze this input:\n<Speech><SpeechHere></Speech>\nOutput:
        # NOT repeating the instruction per shot keeps long def-block prompts well
        # under model_max_length (repeating it K+1 times overflowed → truncation).
        instruction_text = self._apply_symbols(instruction, symbol_mappings)
        image_paths: List[str] = []

        # 0-shot: keep the original image-first single block "<image>\n{instruction}"
        # — identical to the non-ICL training format (medfmc_to_llava bakes
        # "<image>\n{instruction}"), so colon 0-shot inference is unchanged.
        if not examples:
            image_paths.append(test_image_path)
            return f"{DEFAULT_IMAGE_TOKEN}\n{instruction_text}", image_paths

        # K-shot (ICI-style): instruction/definitions ONCE, then compact example
        # pairs (image + label), then the query image.
        parts: List[str] = [instruction_text, "Here are few examples to learn from:"]
        for ex in examples:
            ex_label = self._map_label(ex["label"], symbol_mappings)
            parts.append(f"{DEFAULT_IMAGE_TOKEN}\nAnswer: {ex_label}")
            image_paths.append(ex["image"])
        parts.append("Now analyze this image:")
        parts.append(DEFAULT_IMAGE_TOKEN)
        image_paths.append(test_image_path)

        # Image-token order == image_paths order: [example_0, ..., example_k, test]
        prompt_str = "\n\n".join(parts)
        return prompt_str, image_paths

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_symbols(
        self,
        text: str,
        symbol_mappings: Optional[Dict[str, str]],
    ) -> str:
        """
        Two-pass symbol replacement to prevent cascade corruption.

        Mirrors ICI SymbolManager._apply_mapping_safe():
          Pass 1 — replace each original label with a unique placeholder.
          Pass 2 — replace each placeholder with its target symbol.

        This ensures that if label A maps to a string that contains label B,
        label B is not double-replaced.
        """
        if not symbol_mappings:
            return text

        placeholders: Dict[str, str] = {}
        result = text

        # Length-sorted, whole-word, case-insensitive — matches
        # SymbolManager._apply_mapping_safe() and ICI exactly, so the def block
        # and labels are not corrupted by substring matches.
        sorted_items = sorted(
            symbol_mappings.items(), key=lambda kv: len(kv[0]), reverse=True
        )

        # Pass 1: originals → placeholders (whole-word, case-insensitive)
        for idx, (src, dst) in enumerate(sorted_items):
            placeholder = f"__SYM_{idx}__"
            placeholders[placeholder] = dst
            result = re.compile(
                r"\b" + re.escape(src) + r"\b", re.IGNORECASE
            ).sub(placeholder, result)

        # Pass 2: placeholders → symbols
        for placeholder, dst in placeholders.items():
            result = result.replace(placeholder, dst)

        return result

    def _map_label(
        self,
        label: str,
        symbol_mappings: Optional[Dict[str, str]],
    ) -> str:
        """
        Return the symbol for a label, or the label itself if no mapping.

        For multi-label labels (comma-separated, e.g. "effusion, nodule"),
        each component is mapped independently and rejoined.
        """
        if not symbol_mappings:
            return label
        if "," in label:
            parts = [p.strip() for p in label.split(",")]
            return ", ".join(symbol_mappings.get(p, p) for p in parts)
        return symbol_mappings.get(label, label)
