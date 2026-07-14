import os
import glob
import random
from datasets import Dataset, Audio

# --- Configuration ---
RAW_DATA_DIR = "/home/anmola/data/raw/ravdess_song" 
OUTPUT_BASE_DIR = "/home/anmola/data/data"
SAMPLING_RATE = 16000 

# Note: Disgust and Surprise are excluded from the Song dataset
EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful"
}
STATEMENT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}

def parse_ravdess_song_filename(filepath):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('-')
    
    # Check for 7 parts and ensure it's a song file (Vocal channel == '02')
    if len(parts) != 7 or parts[1] != "02": 
        return None
        
    return {
        "id": name_without_ext,
        "audio_path": filepath,
        "emotion_label": EMOTION_MAP.get(parts[2], "unknown"),
        "text": STATEMENT_MAP.get(parts[4], "unknown statement"),
        "actor_id": int(parts[6])
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
    
    all_data = [parse_ravdess_song_filename(f) for f in wav_files if parse_ravdess_song_filename(f)]
    all_data = [d for d in all_data if d["emotion_label"] != "unknown"]
            
    # Split by Actor
    train_data = [d for d in all_data if d["actor_id"] <= 18]
    val_data = [d for d in all_data if 19 <= d["actor_id"] <= 21]
    test_data = [d for d in all_data if d["actor_id"] >= 22]
    
    print("Converting to Hugging Face datasets...")
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
    train_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "ravdess_song_train_5fewshots"))
    val_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "ravdess_song_validation_5fewshots"))
    test_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "ravdess_song_test_5fewshots"))
    
    train_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "ravdess_song_train_audio_lookup"))
    val_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "ravdess_song_validation_audio_lookup"))
    test_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "ravdess_song_test_audio_lookup"))
    
    print("Done! RAVDESS Song is ready for training.")

if __name__ == "__main__":
    main()