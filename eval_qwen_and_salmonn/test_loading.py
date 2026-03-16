import sys, torch, json
sys.path.insert(0, '/home/harinis/ICL_qwen_run/SALMONN')
from config import Config
from models.salmonn_org import SALMONN
from utils import prepare_one_sample
from transformers import WhisperFeatureExtractor
import argparse

args = argparse.Namespace(
    cfg_path='/home/harinis/ICL_qwen_run/SALMONN/configs/decode_config.yaml',
    device='cuda:0',
    options=None,
    lora_rank=64,
    lora_alpha=128,
    lora_dropout=0.1,
    lora_checkpoint='/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2602_1609_orchestrator_5e_10sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt'
)

cfg = Config(args)
cfg.config.model.lora_rank = 64
cfg.config.model.lora_alpha = 128

ckpt_path = cfg.config.model.get("ckpt", "")
cfg.config.model.ckpt = ""
model = SALMONN.from_config(cfg.config.model)
cfg.config.model.ckpt = ckpt_path
ckpt = torch.load(ckpt_path, map_location="cpu")
non_lora = {k: v for k, v in ckpt["model"].items() if "lora" not in k}
model.load_state_dict(non_lora, strict=False)

ckpt2 = torch.load(args.lora_checkpoint, map_location='cpu', weights_only=False)
sd = {(k[len('salmonn.'):] if k.startswith('salmonn.') else k): v for k, v in ckpt2['model_state'].items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"Loaded {len(sd)} keys. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
model.to(args.device).eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

line = open('data/asr/librispeech_test_other_only.jsonl').readline()
data = json.loads(line)
audio_path = data['audio']
print(f"Audio: {audio_path}")
print(f"Ground truth: {data['gt']}")

samples = prepare_one_sample(audio_path, wav_processor)
samples = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in samples.items()}

# TEST 1: ASR prompt
prompt_asr = cfg.config.model.prompt_template.format(
    "<Speech><SpeechHere></Speech> Recognize the speech and give me the transcription."
)

# TEST 2: EXACT prompt from your training log (epoch 0 symbols)
# NOTE: symbols are dynamic per epoch - epoch 1 checkpoint may have different symbols
# but the prompt STRUCTURE is correct
prompt_cls = cfg.config.model.prompt_template.format(
    """<Speech><SpeechHere></Speech> You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- obomp: Shows understanding or receipt of information
- ainq: Expresses agreement
- humo: Expresses disagreement
- raid: General response to a question
- mmbt: Expression of regret or sorry
- usid: Brief verbal/textual feedback (like "uh-huh", "mm-hmm")
- wycra: Speech repairs, repetitions, or corrections
- eqkt: Actions that don't fit other categories
- zlog: Questions to verify understanding
- iyar: General information-seeking questions
- loat: Requests for repetition
- knak: Self-directed speech
- iiej: Concluding statements
- eerga: General statements or information
- njhw: Instructions or directions
- ppdd: Opening statements or greetings
- npot: Statements describing issues or problems
- jele: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions

Here are few examples to learn from:
Text: um
Output: wycra, eqkt

Text: okay um
Output: obomp, wycra

Now analyze this input:
Output:"""
)

with torch.cuda.amp.autocast(dtype=torch.float16):
    out_asr = model.generate(samples, cfg.config.generate, prompts=prompt_asr)[0]
    out_cls = model.generate(samples, cfg.config.generate, prompts=prompt_cls)[0]

print(f"\nASR prompt output:            {out_asr}")
print(f"Classification prompt output: {out_cls}")
print(f"\nExpected: some symbol like 'wycra', 'npot', 'eerga' etc")
print(f"If classification gives a symbol -> model loaded correctly")
print(f"If classification gives random text -> model loading issue")