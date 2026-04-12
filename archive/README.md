# **README.md - Unified Symbol Discovery and LoRA Training System**

## **Overview**
This system implements a novel approach to multimodal AI training that automatically discovers meaningful symbol representations through alternating LoRA (Low-Rank Adaptation) and MLP (Multi-Layer Perceptron) training phases. The core innovation is learning to transform random symbols into semantically meaningful tokens that improve model performance on speech understanding tasks.

---

## **Core Problem Being Solved**

Traditional multimodal models struggle with **symbol grounding** - connecting abstract labels with real-world concepts. When we have meaningful original symbols like "alpha", "beta", "gamma", the model might develop biases based on these human-chosen labels.

This system addresses this by:
1. **Starting with random symbols** to replace meaningful originals (e.g., "alpha" → "duh")
2. **Learning transformations** that map random symbols to meaningful vocabulary tokens
3. **Progressively refining** these mappings through multiple training cycles
4. **Discovering interpretable representations** that reveal what the model has truly learned

---

## **System Architecture**

### **MLPSalmonn Model Components**

#### **Base Foundation**
- **SALMONN Core**: Speech-language multimodal model combining Whisper (speech) and LLaMA (language)
- **LoRA Integration**: Low-rank adaptation for efficient fine-tuning of the language model
- **Position-wise MLP**: Learnable transformation module for symbol discovery

#### **Symbol Transformation Pipeline**
1. **Input Processing**: Speech audio + text prompts containing random symbol placeholders
2. **MLP Transformation**: Converts random symbol embeddings into transformed representations
3. **Vocabulary Attention**: Maps transformed embeddings to actual vocabulary tokens
4. **Soft/Hard Quantization**: Differentiable training vs discrete inference

### **Key Innovation: Dual Quantization Strategy**

#### **Training Mode (Soft Quantization)**
- Uses **temperature-controlled softmax** over entire vocabulary
- Creates **weighted combinations** of vocabulary embeddings
- Maintains **differentiability** for gradient flow
- Enables **end-to-end learning**

#### **Inference Mode (Hard Quantization)**
- Uses **argmax selection** for discrete token discovery
- Provides **interpretable mappings** (random symbol → discovered token)
- Ensures **computational efficiency** during deployment
- Reveals **learned associations**

---

## **Training Process Flow**

### **Phase 1: Initialization**

#### **Dataset Setup**
1. Load multiple speech understanding datasets
2. Extract meaningful original symbols (e.g., "alpha", "beta", "gamma", "delta")
3. Create combined dataloader with balanced sampling

#### **Random Symbol Generation**
1. Generate random 4-5 character words (e.g., "duh", "kesh", "zolf", "plix")
2. Verify each word tokenizes to exactly 2 tokens
3. Create initial mapping: `{"alpha": "duh", "beta": "kesh", "gamma": "zolf", "delta": "plix"}`
4. Replace all meaningful symbols with these random symbols in training data

#### **Model Initialization**
1. Load base SALMONN model with speech and language capabilities
2. Initialize position-wise MLP for symbol transformation
3. Configure LoRA adapters for efficient fine-tuning
4. Set up tracking for random symbol token IDs

### **Phase 2: Alternating Training Cycles**

The system runs multiple cycles, each containing two training phases:

#### **LoRA Training Phase**
**Purpose**: Adapt the language model to work with current symbol mappings

**Process**:
1. **Freeze MLP weights** - prevent symbol transformations from changing
2. **Unfreeze LoRA weights** - allow language model adaptation
3. **Apply current symbol mappings** to all training text
4. **Train with transformed symbols** in prompts and completions
5. **Optimize LoRA parameters** to associate symbols with correct outputs

**What the model learns**:
- How to interpret current symbol representations (whether random or discovered)
- Associations between transformed symbols and expected outputs
- Context-dependent symbol usage patterns

#### **MLP Training Phase**
**Purpose**: Learn better symbol transformations based on current model understanding

**Process**:
1. **Freeze LoRA weights** - maintain current language understanding
2. **Unfreeze MLP weights** - allow symbol transformations to change
3. **Apply symbol mappings** to training data
4. **Train MLP to minimize prediction loss** through symbol transformation
5. **Use soft quantization** for differentiable learning

**What the model learns**:
- Which vocabulary tokens are most useful for current random symbols
- How to transform random symbols toward meaningful representations
- Gradients flow from task loss back to symbol transformations

### **Phase 3: Symbol Discovery**

#### **Discovery Process**
1. **Switch to evaluation mode** for hard quantization
2. **Extract current random symbol token embeddings**
3. **Apply learned MLP transformations**
4. **Compute cosine similarities** with entire vocabulary
5. **Select best English tokens** (excluding originals, above similarity threshold)
6. **Create token-level mappings**: `{random_token_id: meaningful_token_id}`

#### **Symbol Mapping Conversion**
1. **Decode discovered tokens** back to text representations
2. **Handle multi-token symbols** by processing each component
3. **Create new symbol mappings**: `{"alpha": "speaker", "beta": "voice", "gamma": "audio", "delta": "person"}`
4. **Update model's symbol tracking** for next cycle

#### **Discovery Outputs**
- **Token discovery JSON**: Detailed similarities and transformations
- **Symbol mapping JSON**: High-level symbol evolution
- **Logging**: Interpretable symbol changes with confidence scores

---

## **Progressive Symbol Evolution**

### **Cycle-by-Cycle Progression**

#### **Cycle 0: Random Initialization**
- **Original symbols**: `["alpha", "beta", "gamma", "delta"]`
- **Random replacements**: `{"alpha": "duh", "beta": "kesh", "gamma": "zolf", "delta": "plix"}`
- **Model state**: No meaningful associations, working with random strings
- **Training focus**: Learning basic symbol-output relationships with random symbols

#### **Cycle 1: First Discovery**
- **LoRA training**: Adapts to random symbols ("duh", "kesh", etc.)
- **MLP training**: Learns transformations from random symbols toward useful tokens
- **Discovery result**: `{"alpha": "voice", "beta": "speaker", "gamma": "audio", "delta": "person"}`
- **Progress**: Random symbols → speech-related concepts

#### **Cycle 2: Refinement**
- **Training with**: More meaningful discovered symbols ("voice", "speaker", "audio", "person")
- **Better performance**: Model understands speech concepts better
- **Discovery result**: `{"alpha": "speech", "beta": "audio", "gamma": "sound", "delta": "human"}`
- **Progress**: Further semantic refinement and specialization

#### **Final Cycle: Convergence**
- **Stable symbols**: Converged to optimal representations
- **Best performance**: Model effectively uses discovered symbols
- **Interpretable mapping**: Reveals learned conceptual associations

### **Why This Approach Works**

#### **Gradient Flow Alignment**
- **LoRA phase**: Gradients flow from task loss to symbol understanding
- **MLP phase**: Gradients flow from task loss to symbol transformations
- **Discovery phase**: Hard quantization reveals optimal transformations
- **Cycle iteration**: Progressively improves symbol-concept alignment

#### **Unbiased Semantic Grounding**
- **Random start**: Prevents bias toward human-chosen meaningful labels
- **Learned associations**: Model discovers its own optimal representations
- **Vocabulary constraint**: Ensures symbols map to real language concepts
- **Progressive refinement**: Each cycle builds on previous discoveries

---

## **Expected Outcomes and Applications**

### **Symbol Evolution Examples**
```
Initial:    {"alpha": "duh",    "beta": "kesh",   "gamma": "zolf",  "delta": "plix"}
                ↓                    ↓                 ↓                ↓
Cycle 1:    {"alpha": "voice",  "beta": "speaker", "gamma": "audio", "delta": "person"}
                ↓                    ↓                 ↓                ↓
Cycle 2:    {"alpha": "speech", "beta": "audio",   "gamma": "sound", "delta": "human"}
                ↓                    ↓                 ↓                ↓
Final:      {"alpha": "vocal",  "beta": "speaker", "gamma": "audio", "delta": "person"}
```

### **Quantitative Results**
- **Improved accuracy**: Better task performance with discovered symbols
- **Faster convergence**: More efficient learning with meaningful representations
- **Reduced overfitting**: Better generalization through symbol grounding

### **Qualitative Insights**
- **Interpretable discoveries**: Understand what concepts the model learns
- **Cross-dataset patterns**: See how symbols generalize across tasks
- **Semantic evolution**: Track progression from random to meaningful

---

## **System Advantages**

### **Novel Contributions**
1. **Automatic symbol discovery**: No manual label engineering required
2. **Unbiased learning**: Starts from random symbols to avoid human bias
3. **Differentiable pipeline**: End-to-end optimization of symbol representations
4. **Progressive refinement**: Iterative improvement through multiple cycles
5. **Interpretable results**: Clear mapping from random to learned concepts

### **Practical Benefits**
1. **Reduced human bias**: Model discovers its own optimal representations
2. **Better performance**: Learned symbols improve task accuracy
3. **Transferable insights**: Discovered symbols reveal model understanding
4. **Efficient training**: Alternating phases optimize different aspects separately

This system represents a significant advance in learning interpretable, task-optimized symbol representations for multimodal AI systems, with applications spanning speech understanding, domain adaptation, and interpretable machine learning.

---

# **README_SYMBOL.md - Symbol Discovery and Evolution Details**

## **Symbol Transformation Mechanics**

### **Random Symbol Generation Process**

#### **Why 2-Token Symbols?**
- **Complexity Balance**: More complex than single tokens but still manageable
- **Rich Representations**: Allows for richer semantic representations
- **Discovery Efficiency**: Easier to discover meaningful mappings during training
- **Computational Feasibility**: Not too complex for gradient-based optimization

#### **Generation Algorithm**
```
1. Generate random 4-5 character lowercase words
2. Encode with tokenizer to check token count
3. Keep only words that produce exactly 2 tokens
4. Verify decoding matches original word
5. Ensure no overlap with existing vocabulary
```

#### **Example Generation**
```
Attempts: "abcd", "efgh", "ijkl", "mnop", "qrst", "duh", "kesh"
Valid 2-token words: "duh" → [1234, 5678], "kesh" → [9876, 5432]
Final selection: ["duh", "kesh", "zolf", "plix"]
```

### **Symbol Mapping Evolution**

#### **Initial State (Cycle 0)**
```json
{
  "original_to_random": {
    "alpha": "duh",
    "beta": "kesh", 
    "gamma": "zolf",
    "delta": "plix"
  },
  "training_data_transformation": {
    "before": "Classify this alpha speech sample",
    "after": "Classify this duh speech sample"
  }
}
```

#### **MLP Learning Process**
1. **Input**: Random symbol embeddings for "duh" tokens [1234, 5678]
2. **MLP Transformation**: `transformed = original + 0.2 * mlp(original)`
3. **Similarity Computation**: Cosine similarity with all vocabulary embeddings
4. **Soft Quantization**: Temperature-controlled softmax over vocabulary
5. **Gradient Flow**: Loss gradients guide MLP to useful transformations

#### **Discovery Process Detail**
```
Random symbol "duh" (tokens [1234, 5678]):
  → MLP transforms embeddings
  → Compute similarities with vocabulary
  → Top candidates: "voice"(0.87), "audio"(0.84), "sound"(0.79)
  → Select "voice" as best match
  → Update mapping: "alpha": "duh" → "alpha": "voice"
```

### **Symbol Evolution Patterns**

#### **Typical Evolution Trajectory**
```
Stage 1 - Random Baseline:
  {"alpha": "duh", "beta": "kesh", "gamma": "zolf", "delta": "plix"}
  Performance: Poor (model confused by random symbols)

Stage 2 - First Discovery (Speech Domain):
  {"alpha": "voice", "beta": "speaker", "gamma": "audio", "delta": "sound"}
  Performance: Improved (basic speech concepts discovered)

Stage 3 - Refinement (Specialized):
  {"alpha": "vocal", "beta": "person", "gamma": "speech", "delta": "human"}
  Performance: Better (more specialized and accurate concepts)

Stage 4 - Convergence (Optimal):
  {"alpha": "speech", "beta": "speaker", "gamma": "audio", "delta": "person"}
  Performance: Best (stable, task-optimal representations)
```

#### **Cross-Dataset Generalization**
```
VoxCeleb Dataset Discovery:
  "alpha" → "speaker" (person identification focus)
  "beta" → "voice" (vocal characteristics focus)

HVB Dataset Discovery:
  "alpha" → "audio" (acoustic signal focus) 
  "beta" → "speech" (language content focus)

Combined Learning:
  Final symbols balance both perspectives
```

### **Discovery Output Format**

#### **Token-Level Discovery (cycle_N_tokens.json)**
```json
{
  "timestamp": "2024-06-05T10:30:00",
  "total_tokens": 8,
  "changed_tokens": 6,
  "discoveries": [
    {
      "original_token_id": 1234,
      "original_text": "du",
      "discovered_token_id": 9876,
      "discovered_text": "vo",
      "similarity": 0.87
    },
    {
      "original_token_id": 5678, 
      "original_text": "h",
      "discovered_token_id": 5432,
      "discovered_text": "ice",
      "similarity": 0.84
    }
  ]
}
```

#### **Symbol-Level Mapping (cycle_N_symbols.json)**
```json
{
  "timestamp": "2024-06-05T10:30:00",
  "total_symbols": 4,
  "changed_symbols": 3,
  "symbol_changes": [
    {
      "original_label": "alpha",
      "current_symbol": "duh",
      "discovered_symbol": "voice",
      "change_type": "CHANGED",
      "token_mapping": {"1234": 9876, "5678": 5432}
    }
  ],
  "final_mappings": {
    "alpha": "voice",
    "beta": "speaker", 
    "gamma": "audio",
    "delta": "sound"
  }
}
```

### **Technical Implementation Details**

#### **MLP Architecture**
```
Input: Random symbol embeddings [batch_size, embed_dim]
    ↓
Linear(embed_dim → hidden_dim)  # Usually 8 or 16
    ↓
LayerNorm(hidden_dim)  # Stability
    ↓
GELU()  # Smooth activation
    ↓
Dropout(0.1)  # Regularization
    ↓
Linear(hidden_dim → embed_dim)
    ↓
Residual: output = input + 0.2 * mlp_output
```

#### **Soft Quantization Formula**
```
transformed = normalize(original + scale * mlp(original))
similarities = cosine_similarity(transformed, vocab_embeddings)
soft_weights = softmax(similarities / temperature)
final_embedding = weighted_sum(soft_weights, vocab_embeddings)
```

#### **Hard Quantization (Discovery)**
```
similarities = cosine_similarity(transformed, vocab_embeddings)
best_token_id = argmax(similarities)
discovered_embedding = vocab_embeddings[best_token_id]
```

### **Symbol Quality Metrics**

#### **Similarity Thresholds**
- **Minimum similarity**: 0.3 (below this, keep original symbol)
- **High confidence**: >0.8 (strong discovery)
- **Medium confidence**: 0.5-0.8 (reasonable discovery)
- **Low confidence**: 0.3-0.5 (weak but acceptable)

#### **Discovery Success Patterns**
```
Successful Discovery Indicators:
- High cosine similarity (>0.7)
- Semantically relevant to task domain
- Stable across multiple cycles
- Improves downstream task performance

Failed Discovery Indicators:
- Low similarity (<0.3)
- Semantically unrelated tokens
- Frequent changes between cycles
- No performance improvement
```

### **Debugging and Analysis**

#### **Symbol Evolution Tracking**
```
Cycle 0: alpha="duh" → Similarity to "voice"=0.23 (below threshold)
Cycle 1: alpha="duh" → Similarity to "voice"=0.78 (discovered!)
Cycle 2: alpha="voice" → Similarity to "speech"=0.82 (refined!)
Cycle 3: alpha="speech" → Similarity to "speech"=0.99 (converged)
```

#### **Common Discovery Patterns**
```
Audio/Speech Domain:
  Random → {voice, audio, speech, sound, vocal, speaker, person}

Visual Domain (if extended):
  Random → {image, visual, picture, scene, object, color}

Text Domain:
  Random → {text, word, language, sentence, document, content}
```

### **System Robustness Features**

#### **Discovery Validation**
- **English token filtering**: Only discover actual English words
- **Vocabulary range limiting**: Avoid special tokens and rare words
- **Stability checking**: Reject rapidly changing discoveries
- **Performance validation**: Monitor task accuracy with new symbols

#### **Fallback Mechanisms**
- **No discovery fallback**: Keep previous symbols if no good candidates
- **Partial discovery**: Update only confident discoveries
- **Reset capability**: Return to working symbols if performance drops
- **Manual override**: Allow human intervention if needed

This symbol discovery system provides a robust, interpretable way to learn task-optimal representations while maintaining full transparency about what the model has learned.