import argparse
import torch
import os
import json
import logging
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import sys

# --- SPRInT MODIFICATION: PATH ALIGNMENT ---
# Ensures the script can find the SymbolManager in your project folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from models.symbolAdapter.symbol_manager import SymbolManager
except ImportError:
    # Fallback path if running from subdirectories
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    from models.symbolAdapter.symbol_manager import SymbolManager

def eval_model(args):
    # 1. Environment & Model Initialization
    torch.manual_seed(42) # For reproducibility
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    print(f"🔄 Loading model: {model_name}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, 
        model_base=args.model_base, 
        model_name=model_name
    )
    
    # 2. SYNC LOGIC: Initialize and Load Symbol Manager
    # We use original_labels=["0", "1"] to match MedFMC ColonPath ground truth
    symbol_manager = SymbolManager(
        original_labels=["0", "1"],
        tokenizer=tokenizer,
        dynamic_per_epoch=False,
        symbol_type=args.strategy
    )

    if args.strategy != "regular":
        # CRITICAL: Always load the mappings used during training
        # Priority: --symbol-mappings arg > symbol_mappings.json inside model dir > auto-generate (zero-shot only)
        if args.symbol_mappings and os.path.exists(args.symbol_mappings):
            symbol_manager.load_mappings(args.symbol_mappings)
            print(f"✅ Loaded symbols from: {args.symbol_mappings}")
            print(f"✅ Synchronized Symbols: {symbol_manager.get_current_symbols()}")
        else:
            mapping_path = os.path.join(model_path, "symbol_mappings.json")
            if os.path.exists(mapping_path):
                symbol_manager.load_mappings(mapping_path)
                print(f"✅ Synchronized Symbols: {symbol_manager.get_current_symbols()}")
            else:
                print(f"⚠️  WARNING: symbol_mappings.json not found in {model_path}")
                print(f"⚠️  No --symbol-mappings path provided either.")
                print(f"⚠️  Auto-generating fresh symbols for zero-shot SS-FT inference.")
                print(f"⚠️  NOTE: These will NOT match a fine-tuned model — use only for pipeline testing.")
                auto_save = os.path.join(model_path, "symbol_mappings_autogen.json")
                symbol_manager.save_mappings(auto_save)
                print(f"✅ Auto-generated Symbols: {symbol_manager.get_current_symbols()}")
                print(f"💾 Saved to: {auto_save} (reuse with --symbol-mappings for reproducibility)")

    # 3. Load MedFMC Dataset
    data_path = os.path.expanduser(args.question_file)
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Dataset file not found at {data_path}")
        return

    questions = json.load(open(data_path, "r"))
    print(f"📋 Loaded {len(questions)} samples from {args.question_file}")

    results = []
    correct_count = 0

    # 4. Inference Loop
    print(f"🚀 Starting MedFMC Inference (Mode: {args.strategy})...")
    for line in tqdm(questions):
        image_file = line["image"]
        # Ground truth is usually '0' or '1' string
        ground_truth = str(line["conversations"][1]["value"]).strip()
        human_prompt = line["conversations"][0]["value"]
        
        # Prepare Prompt Template
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], human_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Robust Path Handling
        img_full_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(img_full_path):
            print(f"⚠️ Missing Image: {img_full_path}")
            continue

        try:
            image = Image.open(img_full_path).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # Generation Settings
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    max_new_tokens=15, # Increased slightly to catch multi-token symbols
                    use_cache=True
                )
            
            # 5. POST-PROCESSING & CLEANING
            raw_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # STRIP LLaVA Template leftovers
            processed_output = raw_output.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw_output else raw_output
            
            # SYMBOLIC DECODE: 'kxzy' -> '1'
            final_prediction = symbol_manager.convert_symbols_back(processed_output).strip()

            # Accuracy Check
            is_correct = (final_prediction == ground_truth)
            if is_correct:
                correct_count += 1
                
            results.append({
                "id": line.get("id", "N/A"),
                "image": image_file,
                "raw_output": raw_output,
                "decoded_pred": final_prediction,
                "gt": ground_truth,
                "correct": is_correct
            })

        except Exception as e:
            print(f"⚠️ Error processing sample {line.get('id')}: {e}")
            continue

    # 6. Final MedFMC Accuracy Reporting
    total = len(results)
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    
    print("\n" + "="*50)
    print(f"📊 FINAL MEDFMC RESULTS | STRATEGY: {args.strategy.upper()}")
    print(f"📍 Checkpoint: {os.path.basename(model_path)}")
    print(f"✅ Correct:    {correct_count} / {total}")
    print(f"📈 Accuracy:   {accuracy:.2f}%")
    print("="*50 + "\n")

    # Save outputs for paper/thesis verification
    out_name = f"metrics_{args.strategy}_{os.path.basename(model_path)}.json"
    with open(out_name, "w") as f:
        json.dump({"accuracy": accuracy, "details": results}, f, indent=2)
    print(f"💾 Results saved to: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-path", type=str, required=True, help="Path to your LoRA checkpoint")
    parser.add_argument("--image-folder", type=str, required=True, help="Root folder for MedFMC images")
    parser.add_argument("--question-file", type=str, required=True, help="Path to test.json or val.json")
    parser.add_argument("--strategy", type=str, default="two_token", choices=["regular", "two_token"])
    parser.add_argument("--symbol-mappings", type=str, default=None,
                        help="Path to symbol_mappings.json (SS-FT only). Overrides auto-detection inside --model-path.")
    args = parser.parse_args()
    
    eval_model(args)