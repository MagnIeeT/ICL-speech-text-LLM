import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import re
import sys
from evaluate_tokenizer import EvaluationTokenizer
import editdistance as ed
import torch
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from cn_tn import TextNorm
import zhconv

english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
        to_banjiao = False,
        to_upper = False,
        to_lower = False,
        remove_fillers = False,
        remove_erhua =False,
        check_chars = False,
        remove_space = False,
        cc_mode = '',
    )
basic_normalizer = BasicTextNormalizer()

from tqdm import tqdm

from pathlib import Path
import sys

# Add project root and SALMONN root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
SALMONN_ROOT = PROJECT_ROOT.parent / "SALMONN"
sys.path.append(str(SALMONN_ROOT))

# CHANGE 1: SALMONN imports instead of Qwen
from config import Config
from models.salmonn_org import SALMONN
from utils import prepare_one_sample
from transformers import WhisperFeatureExtractor

PUNCS = '!,.?;:'

ds_collections = {
    'librispeech': {'path': 'data/asr/librispeech_test_clean_only.jsonl','language': 'en'},
    'librispeech_other': {'path': 'data/asr/librispeech_test_other_only.jsonl','language': 'en'},
    'aishell2': {'path': 'asr/aishell2_eval.jsonl', 'language': 'zh'},
    'cv15_en': {'path': 'asr/cv15_asr_en_eval.jsonl', 'language': 'en'},
    'cv15_zh': {'path': 'asr/cv15_asr_zh_eval.jsonl', 'language': 'zh'},
    'cv15_yue': {'path': 'asr/cv15_asr_yue_eval.jsonl', 'language': 'yue'},
    'cv15_fr': {'path': 'asr/cv15_asr_fr_eval.jsonl', 'language': 'fr'},
    'fluers_zh': {'path': 'asr/fleurs_asr_zh_eval.jsonl', 'language': 'zh'},
}

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        path = ds['path']
        self.datas = open(path).readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['audio']
        source = data['source']
        gt = data['gt']

        return {
            'audio': audio,
            'source': source,
            'gt': gt
        }

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt

def compute_wer(refs, hyps, language):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
            tokenizer_type="none",
            lowercase=True,
            punctuation_removal=True,
            character_tokenization=False,
        )
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]
        if language in ["yue"]:
            ref = zhconv.convert(ref, 'zh-cn')
            pred = zhconv.convert(pred, 'zh-cn')
        if language in ["en"]:
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        if language in ["zh"]:
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)
        else:
            ref = basic_normalizer(ref)
            pred = basic_normalizer(pred)
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        if language in ["zh", "yue"]:
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]
        if i==0:
            print(f"ref: {ref}")
            print(f"pred: {pred}")
            print(f"ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}")
            print(f"pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}")
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    return distance/ref_length

def load_model(args):
    cfg = Config(args)
    if args.lora_rank:
        cfg.config.model.lora_rank = args.lora_rank
    if args.lora_alpha:
        cfg.config.model.lora_alpha = args.lora_alpha
    if args.lora_checkpoint:
        # For custom LoRA: load base weights without salmonn_v1 lora weights
        import torch as _torch
        ckpt_path = cfg.config.model.get("ckpt", "")
        cfg.config.model.ckpt = ""
        model = SALMONN.from_config(cfg.config.model)
        cfg.config.model.ckpt = ckpt_path
        if ckpt_path:
            ckpt = _torch.load(ckpt_path, map_location="cpu")
            non_lora = {k: v for k, v in ckpt["model"].items() if "lora" not in k}
            model.load_state_dict(non_lora, strict=False)
    else:
        model = SALMONN.from_config(cfg.config.model)

    if args.lora_checkpoint:
        print(f"Loading LoRA checkpoint: {args.lora_checkpoint}")
        ckpt = torch.load(args.lora_checkpoint, map_location='cpu', weights_only=False)
        sd = {
            (k[len('salmonn.'):] if k.startswith('salmonn.') else k): v
            for k, v in ckpt['model_state'].items()
        }
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loading fine-tuned checkpoint weights...")
        print(f"Loaded {len(sd)} parameters from checkpoint")
        print(f"Missing keys: {len(missing)} (expected for frozen params)")
        print(f"Dataset: {args.dataset}  Total samples: loaded next")

    model.to(args.device)
    return model.eval(), cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True, help='path to config yaml')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--options", nargs="+")
    parser.add_argument('--dataset', type=str, default='librispeech_other')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lora-checkpoint', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--lora-rank', type=int, default=64)
    parser.add_argument('--lora-alpha', type=int, default=128)
    parser.add_argument('--lora-dropout', type=float, default=0.1)

    args = parser.parse_args()

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(random.randint(29500, 29999)))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    torch.distributed.init_process_group(
        backend='gloo',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)  
    print('==========================================')
    print(f'CFG PATH: {args.cfg_path}')
    print(f'DATASET: {args.dataset}')
    print(f'LORA CHECKPOINT: {args.lora_checkpoint}')
    print(f'LORA RANK: {args.lora_rank}  ALPHA: {args.lora_alpha}')
    print('==========================================')
    model, cfg = load_model(args)
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    random.seed(args.seed)
    dataset = AudioDataset(ds=ds_collections[args.dataset])
    if args.num_samples:
        dataset.datas = dataset.datas[:args.num_samples]

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    gts, sources, rets, audio_paths = [], [], [], []
    instruction = "Recognize the speech and give me the transcription."

    for _, batch in tqdm(enumerate(data_loader)):
        audio_path = batch['audio'][0]
        gt = batch['gt'][0]
        source = batch['source'][0]

        try:
            # Prepare samples and move to the model's device
            samples = prepare_one_sample(audio_path, wav_processor)
            samples = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in samples.items()}
            
            # Use official prompt template and interleave speech tag
            prompt = cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + instruction)
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model.generate(samples, cfg.config.generate, prompts=prompt)[0]
            
            # Clean output text
            output = output.replace("<s>", "").replace("</s>", "").strip()
            for prefix in ["the transcription of the given speech is", "the transcription is", "the speech says", "transcription:"]:
                if output.lower().startswith(prefix):
                    output = output[len(prefix):].lstrip(": ").strip()
            
        except Exception as e:
            print(f"Error on {audio_path}: {e}")
            output = ""

        gts.append(gt)
        rets.append(output)
        sources.append(source)
        audio_paths.append(audio_path)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_gts = [None for _ in range(world_size)]
    merged_sources = [None for _ in range(world_size)]
    merged_responses = [None for _ in range(world_size)]
    merged_audio_paths = [None for _ in range(world_size)]
    
    torch.distributed.all_gather_object(merged_gts, gts)
    torch.distributed.all_gather_object(merged_sources, sources)
    torch.distributed.all_gather_object(merged_responses, rets)
    torch.distributed.all_gather_object(merged_audio_paths, audio_paths)

    if torch.distributed.get_rank() == 0:
        merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
        merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
        merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
        merged_responses = [_ for _ in itertools.chain.from_iterable(merged_responses)]

        print(f"Evaluating {args.dataset} ...")
        results = []
        for gt, response, source, audio_path in zip(merged_gts, merged_responses, merged_sources, merged_audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'source': source,
                'audio_path': audio_path,
            })
        
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))
        
        results_dict = {}
        for item in results:
            source = item["source"]
            results_dict.setdefault(source, []).append(item)
            
        lan = ds_collections[args.dataset]['language']
        for source in results_dict:
            refs, hyps = [], []
            for result in results_dict[source]:
                gt = remove_sp(result["gt"], lan)
                response = remove_sp(result["response"], lan)
                refs.append(gt)
                hyps.append(response)
            
            wer = compute_wer(refs, hyps, lan)
            print(f"source: {source}  cnt: {len(refs)} wer: {wer:.4f}")

    torch.distributed.barrier()