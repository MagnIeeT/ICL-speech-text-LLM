import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import sacrebleu
import torch
import requests
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read


ds_collections = {
    'covost2': {'path': 'data/st/covost2_en_de_en_zh_eval.jsonl'}
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
        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>"+data['prompt']
        gt = data['gt']

        return {
            'audio': audio,
            'prompt': prompt,
            'source': source,
            'gt': gt
        }

def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

def collate_fn(inputs, processor):
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio'] for _ in inputs]
    input_audios = [ffmpeg_read(read_audio(_['audio']),sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
    inputs = processor(text=input_texts, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs, audio_path, source, gt


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


def load_model(args):
    """
    Load model based on whether lora_checkpoint is provided or not.
    - No lora_checkpoint: load plain Qwen2-Audio-7B (baseline)
    - lora_checkpoint provided: load base model + apply LoRA weights from checkpoint
    """
    if args.lora_checkpoint:
        print(f"Loading base model: {args.checkpoint}")
        print(f"Applying LoRA checkpoint: {args.lora_checkpoint}")

        from peft import LoraConfig, get_peft_model, TaskType
        from pathlib import Path
        import sys
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(PROJECT_ROOT))

        base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        for param in base_model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=args.lora_dropout,
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
        print(f"Applied LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")

        print(f"Loading checkpoint weights...")
        checkpoint = torch.load(args.lora_checkpoint, map_location='cpu', weights_only=False)

        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[len('model.'):]] = v
                else:
                    new_state_dict[k] = v

            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            loaded_params = len(new_state_dict) - len(unexpected_keys)
            print(f"Loaded {loaded_params} parameters from checkpoint")
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} (expected for frozen params)")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
        else:
            raise ValueError("Checkpoint does not contain 'model_state' key!")

        model = model.eval()
        model = model.cuda()
        print(f"LoRA model loaded successfully!")

    else:
        print(f"Loading baseline model: {args.checkpoint}")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.checkpoint,
            device_map='cuda',
            torch_dtype='auto',
            trust_remote_code=True
        ).eval()
        print(f"Baseline model loaded successfully!")

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lora-checkpoint', type=str, default=None,
                        help='Path to LoRA checkpoint .pt file. If not set, runs baseline.')
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

    model = load_model(args)

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    dataset = AudioDataset(
        ds=ds_collections[args.dataset],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    gts = []
    sources = []
    rets = []
    audio_paths = []
    for _, (inputs, audio_path, source, gt) in tqdm(enumerate(data_loader)):
        inputs['input_ids'] = inputs['input_ids'].to('cuda')
        output_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False)
        output_ids = output_ids[:, inputs.input_ids.size(1):]
        output = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        gts.extend(gt)
        rets.extend(output)
        sources.extend(source)
        audio_paths.extend(audio_path)

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

    merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
    merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
    merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
    merged_responses = [_ for _ in itertools.chain.from_iterable(merged_responses)]

    if torch.distributed.get_rank() == 0:
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
        for item in tqdm(results):
            source = item["source"]
            results_dict.setdefault(source, []).append(item)

        for source in results_dict:
            text_lan = source.split("_")[-2]
            if text_lan == "ja":
                text_lan = "ja-mecab"
            elif text_lan == "zh":
                text_lan = "zh"
            else:
                text_lan = "13a"
            refs, hyps = [], []
            results_list = results_dict[source]
            for result in results_list:
                gt = result["gt"]
                response = result["response"]
                refs.append(gt)
                hyps.append(response)
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=text_lan).score
            print(f"source: {source}  cnt: {len(refs)} bleu score: {bleu:.4f}")

    torch.distributed.barrier()