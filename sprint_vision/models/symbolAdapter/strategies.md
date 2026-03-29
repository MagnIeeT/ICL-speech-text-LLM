# Symbol Training Strategies for ICL Learning

## Problem Statement

When training a model with symbols (e.g., `neutral → xyz`), the model learns to always output symbols regardless of what labels appear in the in-context examples. This breaks the In-Context Learning (ICL) ability where the model should follow the label format from the few-shot examples.

**Goal:** Train the model to be **symbol-agnostic** - it should output whatever label format is shown in the context (symbols OR original labels).

---

## Current Approach (Baseline)

```
Epoch 0, 1, 2 → Symbol Set A: {neutral: xyz, sadness: abc, anger: def}
Epoch 3, 4, 5 → Symbol Set B: {neutral: pqr, sadness: stu, anger: vwx}
```

**Problem:** Model memorizes symbols for 3 epochs, then has to re-learn new mappings. Doesn't learn to read context.

---

## Proposed Strategies

### Strategy 1: Mixed Symbols Within Same Batch ⭐ (Recommended First)

**Idea:** Within a single batch, different samples use different symbol sets.

```
Batch 1:
  Sample 1: few-shot examples use Set A → target: xyz (Set A symbol for neutral)
  Sample 2: few-shot examples use Set B → target: pqr (Set B symbol for neutral)
  Sample 3: few-shot examples use Set A → target: abc (Set A symbol for sadness)
  Sample 4: few-shot examples use Set B → target: stu (Set B symbol for sadness)
```

**Why it helps:**
- Model MUST read the context to know which symbol set is being used
- Same audio can map to different symbols depending on context
- Strongest signal for learning ICL behavior
- Gradients from same batch reinforce "follow the context" behavior

**Implementation:**
```python
def replace_symbols_in_batch(self, batch, epoch, mix_symbols=True):
    if mix_symbols:
        symbol_set_1 = self.get_symbols_for_epoch(epoch)
        symbol_set_2 = self.generate_fresh_symbols()
        
        updated_prompts = []
        updated_completions = []
        
        for i, (prompt, completion) in enumerate(zip(batch['prompt'], batch['completion'])):
            # Randomly choose symbol set for this sample
            use_set_2 = random.random() < 0.5
            symbols = symbol_set_2 if use_set_2 else symbol_set_1
            
            # Replace symbols in both prompt (few-shot) and completion (target)
            updated_prompt = self._replace_labels_with_symbols(prompt, symbols)
            updated_completion = self._replace_labels_with_symbols(completion, symbols)
            
            updated_prompts.append(updated_prompt)
            updated_completions.append(updated_completion)
        
        return {'prompt': updated_prompts, 'completion': updated_completions, ...}
```

**Pros:**
- Strongest ICL learning signal
- Model cannot memorize symbol mappings
- Stable gradients (both sets in same batch)

**Cons:**
- More complex implementation
- Need to ensure few-shot examples and target use SAME symbol set

---

### Strategy 2: Random Symbol Set Per Batch

**Idea:** Each batch uses a randomly chosen symbol set (from pre-generated sets).

```
Batch 1 → Symbol Set A (randomly chosen)
Batch 2 → Symbol Set B (randomly chosen)
Batch 3 → Symbol Set A (randomly chosen)
Batch 4 → Symbol Set B (randomly chosen)
```

**Implementation:**
```python
def _train_epoch(self, step, epoch):
    # Pre-generate symbol sets
    symbol_sets = [
        self.symbol_manager.get_symbols_for_epoch(0),
        self.symbol_manager.generate_fresh_symbols(),
    ]
    
    for batch_idx, batch in enumerate(dataloader):
        # Randomly choose symbol set for this batch
        current_symbols = random.choice(symbol_sets)
        
        updated_batch = self.symbol_manager.replace_symbols_in_batch(
            batch, symbol_override=current_symbols
        )
```

**Pros:**
- Simple implementation
- Model sees variety within epoch

**Cons:**
- Potential gradient instability (batch N learns Set A, batch N+1 learns Set B)
- Model has no memory of previous batch's symbols

---

### Strategy 3: Random Symbol Set Per Epoch (with more sets)

**Idea:** Each epoch randomly picks from a pool of pre-generated symbol sets.

```
Epoch 0 → Symbol Set C (randomly chosen from {A, B, C, D, E})
Epoch 1 → Symbol Set A (randomly chosen)
Epoch 2 → Symbol Set E (randomly chosen)
Epoch 3 → Symbol Set A (randomly chosen)  # Can repeat!
```

**Implementation:**
```python
def _train_epoch(self, step, epoch):
    # Generate pool of symbol sets once
    if not hasattr(self, 'symbol_pool'):
        self.symbol_pool = [self.symbol_manager.generate_fresh_symbols() for _ in range(10)]
    
    # Randomly pick one for this epoch
    epoch_symbols = random.choice(self.symbol_pool)
```

**Pros:**
- Stable within epoch (same symbols for all batches)
- Variety across epochs
- Can revisit same symbols (reinforcement)

**Cons:**
- Weaker ICL signal than per-batch strategies
- Model might still memorize within epoch

---

### Strategy 4: Mixed Original + Symbols Within Batch ⭐⭐ (Strongest ICL)

**Idea:** Some samples in each batch use **original labels**, others use **symbols**.

```
Batch 1:
  Sample 1: few-shot use "neutral, sadness" → target: "neutral"     (ORIGINAL)
  Sample 2: few-shot use "xyz, abc"         → target: "xyz"         (SYMBOLS Set A)
  Sample 3: few-shot use "pqr, stu"         → target: "pqr"         (SYMBOLS Set B)
  Sample 4: few-shot use "neutral, sadness" → target: "sadness"     (ORIGINAL)
```

**Why it helps:**
- Model learns to output BOTH original labels AND symbols
- Strongest signal for "copy the label format from context"
- Prevents catastrophic forgetting of original labels

**Implementation:**
```python
def replace_symbols_in_batch(self, batch, epoch, mix_with_original=True, original_ratio=0.2):
    symbol_set_1 = self.get_symbols_for_epoch(epoch)
    symbol_set_2 = self.generate_fresh_symbols()
    
    updated_prompts = []
    updated_completions = []
    
    for i, (prompt, completion) in enumerate(zip(batch['prompt'], batch['completion'])):
        rand_val = random.random()
        
        if rand_val < original_ratio:
            # Use original labels (no replacement)
            updated_prompts.append(prompt)
            updated_completions.append(completion)
        elif rand_val < original_ratio + 0.4:
            # Use Symbol Set 1
            symbols = symbol_set_1
            updated_prompts.append(self._replace_labels_with_symbols(prompt, symbols))
            updated_completions.append(self._replace_labels_with_symbols(completion, symbols))
        else:
            # Use Symbol Set 2
            symbols = symbol_set_2
            updated_prompts.append(self._replace_labels_with_symbols(prompt, symbols))
            updated_completions.append(self._replace_labels_with_symbols(completion, symbols))
    
    return {'prompt': updated_prompts, 'completion': updated_completions, ...}
```

**Pros:**
- Preserves original label ICL ability
- Strongest "follow context" signal
- Model learns all three: original, Set A, Set B

**Cons:**
- Most complex implementation
- Need to balance ratios carefully

---

## Recommended Experiment Order

### Experiment 1: Mixed Symbols Within Batch (Strategy 1)
- **Rationale:** Simple to implement, strong ICL signal
- **Config:** 50% Set A, 50% Set B within each batch
- **Expected:** Model learns to read context, Fresh/Original metrics don't collapse

### Experiment 2: Mixed Original + Symbols (Strategy 4)
- **Rationale:** If Exp 1 works, add original labels to the mix
- **Config:** 20% Original, 40% Set A, 40% Set B within each batch
- **Expected:** Best of all worlds - model handles any label format

### Experiment 3: Random Per Epoch (Strategy 3)
- **Rationale:** If per-batch is too unstable, try per-epoch
- **Config:** Pool of 5-10 symbol sets, random selection each epoch
- **Expected:** More stable training, moderate ICL improvement

---

## Implementation Checklist

### For Strategy 1 (Mixed Symbols Within Batch):

- [ ] Modify `SymbolManager.replace_symbols_in_batch()` to accept `mix_symbols=True`
- [ ] Generate two symbol sets at batch start
- [ ] Randomly assign each sample to a symbol set
- [ ] Ensure few-shot examples and target use SAME symbol set per sample
- [ ] Add logging to verify mixed symbols are being used

### For Strategy 4 (Mixed Original + Symbols):

- [ ] Add `original_ratio` parameter (default 0.2)
- [ ] Skip symbol replacement for original samples
- [ ] Track ratio of original vs symbol samples per batch
- [ ] Validate that Original mode metrics improve

---

## Validation Metrics to Track

| Metric | What it measures | Target |
|--------|------------------|--------|
| Symbol Mode | Can model output correct symbols when shown symbols in context | > 0.5 |
| Fresh Mode | Can model output NEW symbols it hasn't seen during training | > 0.4 |
| Original Mode | Can model output original labels when shown original labels in context | > 0.4 |

**Success Criteria:**
- All three modes should maintain reasonable accuracy (> 0.3)
- Original mode should NOT collapse to 0 after training
- Fresh mode should be close to Symbol mode (shows true ICL ability)

---

## Debugging Tips

1. **If Original mode collapses to 0:**
   - Model is overfitting to symbols
   - Try Strategy 4 (mix original labels into training)

2. **If Fresh mode is much lower than Symbol mode:**
   - Model is memorizing specific symbols, not learning ICL
   - Try Strategy 1 (mixed symbols within batch)

3. **If training is unstable (loss spikes):**
   - Per-batch symbol switching might be too aggressive
   - Try Strategy 3 (per-epoch switching) or reduce mixing ratio

4. **If all modes are low:**
   - Check if correct parameters are being trained
   - Verify symbol replacement is working correctly
   - Check learning rate (might be too high)