import os
import glob
import random
from datasets import Dataset, Audio

# --- Configuration ---
RAW_DATA_DIR = "/home/anmola/data/raw/cremad"
OUTPUT_BASE_DIR = "/home/anmola/data/data"
SAMPLING_RATE = 16000

# CREMA-D mapping based on filename specs
EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

STATEMENT_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes"
}

def parse_cremad_filename(filepath):
    """Extracts features from the 4-part filename identifier (e.g., 1001_DFA_ANG_XX.wav)."""
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) < 3: 
        return None
        
    try:
        actor_id = int(parts[0])
    except ValueError:
        return None
        
    return {
        "id": name_without_ext,
        "audio_path": filepath,
        "emotion_label": EMOTION_MAP.get(parts[2], "unknown"),
        "text": STATEMENT_MAP.get(parts[1], "Unknown statement"),
        "actor_id": actor_id
    }

def create_hf_dataset(data_list):
    dataset = Dataset.from_list(data_list)
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=SAMPLING_RATE))
    dataset = dataset.rename_column("audio_path", "audio")
    return dataset

def create_audio_lookup(dataset, num_fewshots=5):
    label_indices = {}
    for idx, label in enumerate(dataset["emotion_label"]):
        label_indices.setdefault(label, []).append(idx)
        
    lookup_data = []
    for label, indices in label_indices.items():
        chosen = random.sample(indices, min(num_fewshots, len(indices)))
        for idx in chosen:
            example = dataset[idx]
            lookup_data.append({
                "index": str(example["id"]),
                "audio": example["audio"]
            })
    return Dataset.from_list(lookup_data)

def main():
    wav_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.wav"), recursive=True)
    print(f"Found {len(wav_files)} .wav files.")
    
    all_data = [parse_cremad_filename(f) for f in wav_files if parse_cremad_filename(f)]
    all_data = [d for d in all_data if d["emotion_label"] != "unknown"]
            
    # Split by Actor to prevent data leakage (91 actors total: 1001 to 1091)
    # Train: 1001-1073 (73 actors), Val: 1074-1082 (9 actors), Test: 1083-1091 (9 actors)
    train_data = [d for d in all_data if d["actor_id"] <= 1073]
    val_data = [d for d in all_data if 1074 <= d["actor_id"] <= 1082]
    test_data = [d for d in all_data if d["actor_id"] >= 1083]
    
    print("Converting to Hugging Face datasets (this will use soundfile/librosa)...")
    train_ds = create_hf_dataset(train_data)
    val_ds = create_hf_dataset(val_data)
    test_ds = create_hf_dataset(test_data)
    
    def add_empty_fewshot(example):
        example["few_shot_examples"] = []
        return example
        
    train_ds = train_ds.map(add_empty_fewshot)
    val_ds = val_ds.map(add_empty_fewshot)
    test_ds = test_ds.map(add_empty_fewshot)

    print("Creating audio lookups...")
    train_lookup = create_audio_lookup(train_ds)
    val_lookup = create_audio_lookup(val_ds)
    test_lookup = create_audio_lookup(test_ds)

    print("Saving to disk...")
    train_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "cremad_train_5fewshots"))
    val_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "cremad_validation_5fewshots"))
    test_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "cremad_test_5fewshots"))
    
    train_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "cremad_train_audio_lookup"))
    val_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "cremad_validation_audio_lookup"))
    test_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "cremad_test_audio_lookup"))
    
    print("Done! CREMA-D is ready for training.")

if __name__ == "__main__":
    main()