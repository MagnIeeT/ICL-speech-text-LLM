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

from typing import Dict, List, Optional, Tuple

from llava.constants import DEFAULT_IMAGE_TOKEN


class LLaVAPromptBuilder:
    """
    Builds the human-turn prompt for LLaVA inference.

    Prompt structure (N-shot, symbol strategy shown):

        <image>
        {instruction with symbols}
        Answer: {example_label_as_symbol}

        <image>
        {instruction with symbols}
        Answer: {example_label_as_symbol}

        <image>
        {instruction with symbols}

    For zero-shot (no examples), only the final block is emitted.
    image_paths is always ordered [example_0, example_1, ..., test_image],
    matching the order LLaVA expects for image tensors.
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
        # Apply symbol mapping to the instruction once.
        # The instruction already embeds "(0 for No, 1 for Yes)" so replacing
        # "0"→"kxzy" and "1"→"wxab" produces "(kxzy for No, wxab for Yes)"
        # naturally — identical to how ICI's prompt_template is handled.
        instruction_text = self._apply_symbols(instruction, symbol_mappings)

        image_paths: List[str] = []
        prompt_parts: List[str] = []

        # ── Example blocks ────────────────────────────────────────────────────
        # Mirrors ICI:
        #   "Here are few examples to learn from:\n"
        #   "Text: {text}\nOutput: {label}\n\n"
        # Each example needs its own <image> token so LLaVA pairs it
        # with the correct image tensor.
        for ex in examples:
            ex_label = self._map_label(ex["label"], symbol_mappings)
            block = (
                f"{DEFAULT_IMAGE_TOKEN}\n"
                f"{instruction_text}\n"
                f"Answer: {ex_label}"
            )
            prompt_parts.append(block)
            image_paths.append(ex["image"])

        # ── Test query ────────────────────────────────────────────────────────
        # Mirrors ICI's "Now analyze this input:\n<Speech>...\nOutput:"
        # No "Answer:" here — the model generates it.
        prompt_parts.append(f"{DEFAULT_IMAGE_TOKEN}\n{instruction_text}")
        image_paths.append(test_image_path)

        # Double newline between blocks (same separator ICI uses between examples)
        prompt_str = "\n\n".join(prompt_parts)

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

        # Pass 1: originals → placeholders
        for idx, (src, dst) in enumerate(symbol_mappings.items()):
            placeholder = f"__SYM_{idx}__"
            placeholders[placeholder] = dst
            result = result.replace(src, placeholder)

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
