import json
import os

def convert_to_llava(task_name, input_txt, img_root, output_json, instruction):
    llava_data = []
    
    if not os.path.exists(input_txt):
        print(f"⚠️ Skipping: {input_txt} not found.")
        return

    with open(input_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            # parts[0] is usually the filename
            img_name = parts[0]
            
            # FIX: Usually in MedFMC, the label is the LAST item on the line
            # We use parts[-1] to ensure we get the 0 or 1
            label = parts[-1] 
            
            entry = {
                "id": f"{task_name}_{img_name.split('.')[0]}",
                "image": os.path.join(img_root, img_name),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{instruction}"},
                    {"from": "gpt", "value": label} 
                ]
            }
            llava_data.append(entry)
            
    with open(output_json, 'w') as f:
        json.dump(llava_data, f, indent=2)
    print(f"✅ Generated {output_json} with {len(llava_data)} samples.")

# ==========================================
# RUN CONVERSIONS
# ==========================================

# 1. COLON DATASET
# I added a check for train_20.txt first, but you can change back to train.txt if needed
convert_to_llava(
    task_name="colon",
    input_txt="/home/harinis/MedFM/data/MedFMC/colon/train_20.txt",
    img_root="colon/images", 
    output_json="/home/harinis/LLaVA/sprint_vision/data/colon_train.json",
    instruction="Task: Medical Image Classification. Dataset: Colon Histopathology. Is there a tumor? Answer with the label only (0 for No, 1 for Yes)."
)

# 2. COLON TEST DATASET
# Using val_20.txt since your previous run showed test.txt was missing
convert_to_llava(
    task_name="colon_test",
    input_txt="/home/harinis/MedFM/data/MedFMC/colon/val_20.txt", 
    img_root="colon/images", 
    output_json="/home/harinis/LLaVA/sprint_vision/data/colon_test.json",
    instruction="Task: Medical Image Classification. Dataset: Colon Histopathology. Is there a tumor? Answer with the label only (0 for No, 1 for Yes)."
)