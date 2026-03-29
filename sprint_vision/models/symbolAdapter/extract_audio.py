import soundfile as sf
from datasets import load_from_disk
import os

# 1. Path to your dataset folder (the one with the .arrow files)
# Based on your previous screenshots, this is the correct folder for the dataset dict
dataset_path = "/home/leapers/weights/harinis/ICL-speech-text-LLM/data/hvb_test_50fewshots"

print(f"Loading dataset from {dataset_path}...")
try:
    dataset = load_from_disk(dataset_path)
    
    # 2. Get the first sample
    print("Extracting the first audio sample...")
    sample = dataset[0]
    
    # 3. Get audio data
    # Note: Structure might be sample['audio']['array'] or just sample['audio']
    audio_data = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    
    # 4. Save as a real WAV file
    output_filename = "evidence.wav"
    sf.write(output_filename, audio_data, sr)
    
    print(f"\n✅ SUCCESS! Saved audio to: {os.path.abspath(output_filename)}")
    print("Now use this file path in your prove_failure.py script!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Try pointing to the 'hvb_test_audio_lookup' folder instead if this fails.")