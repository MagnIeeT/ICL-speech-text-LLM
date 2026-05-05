import sys, torch, json
from pathlib import Path
import argparse

# Add project root and SALMONN root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
SALMONN_ROOT = PROJECT_ROOT.parent / "SALMONN"
sys.path.insert(0, str(SALMONN_ROOT))

from config import Config
from models.salmonn_org import SALMONN
from utils import prepare_one_sample
from transformers import WhisperFeatureExtractor

# Attempt to load .env for paths
from utils.environment import setup_environment, get_env_path
setup_environment()

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default=str(SALMONN_ROOT / 'configs/decode_config.yaml'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lora_checkpoint', type=str, default=None)
    args = parser.parse_args()

    print(f"Loading config from: {args.cfg_path}")
    cfg = Config(args)
    cfg.config.model.lora_rank = 64
    cfg.config.model.lora_alpha = 128

    ckpt_path = cfg.config.model.get("ckpt", "")
    if not ckpt_path:
        ckpt_path = get_env_path("SALMONN_CKPT_PATH")
        cfg.config.model.ckpt = ckpt_path

    print(f"Base checkpoint: {ckpt_path}")
    
    # Initialize model
    # Temporarily clear ckpt to load non-lora weights manually if needed
    orig_ckpt = cfg.config.model.ckpt
    cfg.config.model.ckpt = ""
    model = SALMONN.from_config(cfg.config.model)
    cfg.config.model.ckpt = orig_ckpt
    
    if orig_ckpt:
        ckpt = torch.load(orig_ckpt, map_location="cpu")
        non_lora = {k: v for k, v in ckpt["model"].items() if "lora" not in k}
        model.load_state_dict(non_lora, strict=False)

    if args.lora_checkpoint:
        print(f"Loading LoRA checkpoint: {args.lora_checkpoint}")
        ckpt2 = torch.load(args.lora_checkpoint, map_location='cpu', weights_only=False)
        sd = {(k[len('salmonn.'):] if k.startswith('salmonn.') else k): v for k, v in ckpt2['model_state'].items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded {len(sd)} keys. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    model.to(args.device).eval()

    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    # Use relative path for data
    data_path = PROJECT_ROOT / 'data/asr/librispeech_test_other_only.jsonl'
    if not data_path.exists():
        print(f"Warning: data path {data_path} does not exist. Skipping inference test.")
        return

    line = open(data_path).readline()
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

    with torch.cuda.amp.autocast(dtype=torch.float16):
        out_asr = model.generate(samples, cfg.config.generate, prompts=prompt_asr)[0]

    print(f"\nASR prompt output: {out_asr}")

if __name__ == '__main__':
    run_test()
