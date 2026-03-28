import argparse, torch, os, json
from tqdm import tqdm
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_name", default="colon")
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--rebuild_json", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    disable_torch_init()

    VAL_TXT = "/home/harinis/MedFM/data/MedFMC/colon/val_20.txt"
    IMG_DIR = "/home/harinis/MedFM/data/MedFMC/colon/images"
    JSON_PATH = "/home/harinis/LLaVA/sprint_vision/data/colon_val.json"

    # Always rebuild JSON correctly (fixes filename space bug)
    data = []
    with open(VAL_TXT) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            label = line.split()[-1]
            img_name = line[:-(len(label)+1)]
            data.append({
                "id": img_name.replace(".png",""),
                "image": os.path.join(IMG_DIR, img_name),
                "conversations": [
                    {"from":"human","value":"<image>\nIs there a tumor? Answer 0 for No, 1 for Yes."},
                    {"from":"gpt","value":label}
                ]
            })
    with open(JSON_PATH,"w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON built: {len(data)} samples. First image exists: {os.path.exists(data[0]['image'])}")

    if args.num_samples > 0:
        data = data[:args.num_samples]

    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path), device_map="auto"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.run_name}_predictions.jsonl")

    print(f"Running inference on {len(data)} samples...")
    with open(out_file, "w") as f:
        for item in tqdm(data):
            img_path = item["image"]
            if not os.path.exists(img_path):
                print(f"MISSING: {img_path}"); continue

            image = Image.open(img_path).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()

            question = "Is there a tumor? Answer 0 for No, 1 for Yes."
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=10,
                    min_new_tokens=1,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode FULL output then take the answer part after prompt
            full_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            # The answer is after "ASSISTANT:"
            raw = full_text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full_text else full_text.strip()

            raw_lower = raw.lower().strip()
            if raw_lower.startswith("no") or raw_lower == "0":
                prediction = "0"
            elif raw_lower.startswith("yes") or raw_lower == "1":
                prediction = "1"
            else:
                prediction = ""

            result = {
                "id": item["id"],
                "image": img_path,
                "ground_truth": item["conversations"][1]["value"],
                "raw_output": raw,
                "model_prediction": prediction,
                "correct": prediction == item["conversations"][1]["value"]
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

    print(f"Done → {out_file}")

if __name__ == "__main__":
    main()
