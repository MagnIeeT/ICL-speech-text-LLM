import os
import json
import uuid

# The updated, specific path from your data structure
IMAGE_DIR = "/home/harinis/MedFM/data/MedFMC/colon/images"
OUTPUT_JSON = "/home/harinis/LLaVA/colon_inference.json"

PROMPT = "Analyze this medical image of a colon. What abnormalities, if any, are visible?"

def main():
    print(f"Scanning directory: {IMAGE_DIR}")
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = []

    try:
        # The excellent recursive search addition!
        for root, _, files in os.walk(IMAGE_DIR):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    # Use relative path so run_inference.py handles it correctly
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, IMAGE_DIR)
                    image_files.append(rel_path)
                    
    except FileNotFoundError:
        print(f"ERROR: The folder {IMAGE_DIR} does not exist.")
        return

    if not image_files:
        print("No images found! Please check the folder path.")
        return

    print(f"Found {len(image_files)} images. Generating LLaVA JSON...")

    formatted_data = []

    for img in image_files:
        entry = {
            "id": str(uuid.uuid4()),
            "image": img, 
            "conversations": [
                {
                    "from": "human",
                    "value": PROMPT # NO <image> token here! run_inference.py handles it.
                },
                {
                    "from": "gpt",
                    "value": ""
                }
            ]
        }
        formatted_data.append(entry)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Success! Created {OUTPUT_JSON} with {len(formatted_data)} items.")

if __name__ == "__main__":
    main()