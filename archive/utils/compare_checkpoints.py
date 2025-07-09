import torch
import logging

logging.basicConfig(level=logging.INFO)


def compare_checkpoints(original_path, finetuned_path):
    logging.info(f"\nComparing checkpoints:")
    logging.info(f"Original: {original_path}")
    logging.info(f"Finetuned: {finetuned_path}")
    logging.info("-" * 50)
    
    # Load checkpoints
    original_ckpt = torch.load(original_path, map_location="cpu")
    finetuned_ckpt = torch.load(finetuned_path, map_location="cpu")
    
    # Get model state dicts
    original_state = original_ckpt["model"] if "model" in original_ckpt else original_ckpt
    finetuned_state = finetuned_ckpt["model"] if "model" in finetuned_ckpt else finetuned_ckpt
    
    # Compare keys
    original_keys = set(original_state.keys())
    finetuned_keys = set(finetuned_state.keys())
    
    # Find missing keys
    missing_keys = original_keys - finetuned_keys
    extra_keys = finetuned_keys - original_keys
    
    if missing_keys:
        logging.info("\nKeys present in original but missing in finetuned:")
        for key in sorted(missing_keys):
            if key in original_state:
                shape = original_state[key].shape if torch.is_tensor(original_state[key]) else "Non-tensor"
                logging.info(f"- {key} (shape: {shape})")
    
    if extra_keys:
        logging.info("\nExtra keys in finetuned but not in original:")
        for key in sorted(extra_keys):
            if key in finetuned_state:
                shape = finetuned_state[key].shape if torch.is_tensor(finetuned_state[key]) else "Non-tensor"
                logging.info(f"+ {key} (shape: {shape})")
    
    if not missing_keys and not extra_keys:
        logging.info("\nBoth checkpoints have identical keys!")
    
    # Compare sizes
    original_size = sum(p.numel() * p.element_size() for p in original_state.values() if torch.is_tensor(p)) / (1024 ** 2)
    finetuned_size = sum(p.numel() * p.element_size() for p in finetuned_state.values() if torch.is_tensor(p)) / (1024 ** 2)
    
    logging.info(f"\nOriginal checkpoint size: {original_size:.2f} MB")
    logging.info(f"Finetuned checkpoint size: {finetuned_size:.2f} MB")
    logging.info(f"Size difference: {abs(original_size - finetuned_size):.2f} MB")

compare_checkpoints(
    "/home/sriramg/aneeraj/storage/salmonn_v1.pth",
    "./results/trained_models/finetune_llama2_salmon_speech_15e1b/checkpoints/epoch_9_loss_0.1911/model.pt"
)