import torch
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import logging
import os

# Clean up logging
logging.basicConfig(level=logging.INFO)
# Suppress heavy accelerate logs
logging.getLogger("accelerate").setLevel(logging.ERROR)

# =============================================================================
# 1. SETUP & CONFIGURATION (LOW RAM MODE)
# =============================================================================

device = "cpu"
model_id = "Qwen/Qwen2-Audio-7B-Instruct"
# !!! UPDATE THIS PATH !!!
audio_path = "/home/harinis/ICL_qwen_run/ICL-speech-text-LLM/models/symbolAdapter/evidence.wav"

print(f"⏳ Loading model with Disk Offloading... (This prevents crashes but is slower)")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# ✅ CRASH PREVENTION SETTINGS
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",           # Automatically manages memory
    offload_folder="offload",    # Spills excess memory to disk instead of crashing
    offload_state_dict=True,     # Helps with low RAM
    torch_dtype=torch.float32,   # Standard CPU precision
    low_cpu_mem_usage=True,      # Loads incrementally
    trust_remote_code=True
)

print(f"🎧 Loading audio file: {audio_path}")
try:
    audio_array, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
except Exception as e:
    print(f"❌ Error loading audio: {e}")
    exit(1)

# =============================================================================
# 🧪 TEST 1: HVB-LIKE COMPLEXITY
# =============================================================================

print("\n" + "="*80)
print("🧪 TEST 1: HVB-LIKE MULTI-LABEL COMPLEXITY (18 Options)")
print("="*80)

hvb_prompt = """
You are a dialogue analysis expert. Analyze the audio and identify ALL applicable actions.
Output ONLY the corresponding codes separated by commas. Do not output English words.

Mapping:
- acknowledge -> ilmu
- answer_agree -> mekj
- answer_disagree -> pxow
- answer_general -> ulthr
- answer_no -> qzvm
- answer_yes -> rkwn
- backchannel -> styx
- inform_detail -> uvab
- inform_general -> wcpd
- inform_instruction -> xeqf
- question_check -> yghi
- question_general -> zjkl
- question_open -> amnp
- request -> ctuv
- social_apology -> dwxy
- social_greeting -> ezab
- social_thanks -> fcde
- statement_problem -> ghij

Examples:
Audio 1 -> mekj, ilmu
Audio 2 -> ezab
Audio 3 -> pxow, yghi

Now analyze this input:
"""

conversation = [{"role": "user", "content": [
    {"type": "audio", "audio_url": audio_path}, 
    {"type": "text", "text": hvb_prompt}
]}]

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, audios=[audio_array], return_tensors="pt", padding=True, sampling_rate=sr)

# Move inputs to CPU (model handles offloading automatically)
inputs = inputs.to(device)

print("⚡ Running Inference (Please wait... disk usage makes this slow)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=20, 
        return_dict_in_generate=True, 
        output_scores=True
    )

generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
generated_text_1 = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# LOGIT ANALYSIS
first_token_scores = outputs.scores[0][0]
probs = torch.nn.functional.softmax(first_token_scores, dim=-1)
top_k_probs, top_k_indices = torch.topk(probs, 5)

print(f"\n🤖 Output: '{generated_text_1}'")
print("🔍 Top 5 Candidates:")

for i in range(5):
    token_text = processor.tokenizer.decode([top_k_indices[i]])
    token_prob = top_k_probs[i].item() * 100
    print(f"   Rank {i+1}: '{token_text.strip()}' ({token_prob:.2f}%)")


# =============================================================================
# 🧪 TEST 2: DYNAMIC REMAPPING
# =============================================================================

print("\n" + "="*80)
print("🧪 TEST 2: DYNAMIC SYMBOL ADAPTATION")
print("="*80)

map_v1 = {"greeting": "ilmu", "question": "mekj", "statement": "pxow"}
map_v2 = {"greeting": "aaaa", "question": "bbbb", "statement": "cccc"}

for i, mapping in enumerate([map_v1, map_v2], 1):
    print(f"\n--- Epoch {i} Simulation ---")
    dynamic_prompt = f"""
    Analyze the audio. Output the correct code based on this CURRENT mapping:
    - greeting -> {mapping['greeting']}
    - question -> {mapping['question']}
    - statement -> {mapping['statement']}
    
    Output ONLY the code.
    """
    
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio_url": audio_path}, 
        {"type": "text", "text": dynamic_prompt}
    ]}]
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio_array], return_tensors="pt", padding=True, sampling_rate=sr)
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    gen_text = processor.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    
    expected = list(mapping.values())
    print(f"   Mapping: {mapping}")
    print(f"   Output:  '{gen_text}'")
    
    if gen_text in expected:
        print(f"   Result:  ✅ SUCCESS")
    else:
        print(f"   Result:  ❌ FAIL")