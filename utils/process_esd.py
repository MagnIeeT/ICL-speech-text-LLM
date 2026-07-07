import os
import glob
import random
from datasets import Dataset, Audio

# --- Configuration ---
RAW_DATA_DIR = "/home/anmola/data/raw/esd"
OUTPUT_BASE_DIR = "/home/anmola/data/data"
SAMPLING_RATE = 16000

def load_transcriptions(base_dir):
    transcripts = {}
    txt_files = glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        file_id = parts[0].replace('.wav', '').strip()
                        text = parts[1].strip()
                        transcripts[file_id] = text
        except Exception:
            pass
    return transcripts

def parse_esd_path(filepath, transcripts):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    parts = filepath.split(os.sep)
    
    emotion = "unknown"
    for e in ["neutral", "happy", "angry", "sad", "surprise"]:
        if any(e.lower() == p.lower() for p in parts):
            emotion = e
            break
            
    if emotion == "unknown": return None

    try:
        speaker_id = int(name_without_ext.split('_')[0])
    except:
        return None
        
    text = transcripts.get(name_without_ext, "Unknown statement")
    
    return {
        "id": name_without_ext,
        "audio_path": filepath,
        "emotion_label": emotion,
        "text": text,
        "speaker_id": speaker_id
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
    print("Loading transcriptions...")
    transcripts = load_transcriptions(RAW_DATA_DIR)
    
    wav_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.wav"), recursive=True)
    print(f"Found {len(wav_files)} .wav files.")
    
    all_data = [parse_esd_path(f, transcripts) for f in wav_files]
    all_data = [d for d in all_data if d is not None]
            
    # Split by Speaker to prevent data leakage. 1-10: Chinese, 11-20: English
    # Train (16 speakers), Val (2 speakers), Test (2 speakers)
    val_speakers = [9, 19]
    test_speakers = [10, 20]
    
    train_data = [d for d in all_data if d["speaker_id"] not in val_speakers + test_speakers]
    val_data = [d for d in all_data if d["speaker_id"] in val_speakers]
    test_data = [d for d in all_data if d["speaker_id"] in test_speakers]
    
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
    train_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "esd_train_5fewshots"))
    val_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "esd_validation_5fewshots"))
    test_ds.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "esd_test_5fewshots"))
    
    train_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "esd_train_audio_lookup"))
    val_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "esd_validation_audio_lookup"))
    test_lookup.save_to_disk(os.path.join(OUTPUT_BASE_DIR, "esd_test_audio_lookup"))
    
    print("Done! ESD is ready for training.")

if __name__ == "__main__":
    main()