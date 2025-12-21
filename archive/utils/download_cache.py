# Create a file named download_cache.py
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from huggingface_hub import login

# 1. Authenticate
my_token = "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" 
login(token=my_token)
print("Authentication successful.")

# 2. Download Model (Removed 'token' as it's not gated and caused the error)
print("Downloading Model...")
model_name = 'BAAI/llm-embedder'
AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)

# 3. Download Dataset (Use 'use_auth_token' for older datasets library versions)
print("Downloading Dataset...")
try:
    # Try 'token' first, if it fails, the catch block handles it or we use use_auth_token
    load_dataset("asapp/slue", "voxceleb", use_auth_token=my_token)
    print("Download complete! You can now run on the compute node.")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("If you see a 'token' error again, try removing the use_auth_token argument since you already logged in.")