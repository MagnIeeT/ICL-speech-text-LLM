def convert_checkpoint_format(checkpoint_path, model, save_path=None):
    """Convert partial checkpoint to full state dict format"""
    logging.info(f"Loading partial checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Get full model state dict
    full_state_dict = model.state_dict()
    logging.info(f"Full model state dict keys: {len(full_state_dict.keys())}")
    
    # Update with trained parameters
    if 'model_state_dict' in checkpoint:
        trained_state_dict = checkpoint['model_state_dict']
        logging.info("Found 'model_state_dict' in checkpoint")
    elif 'model' in checkpoint:
        trained_state_dict = checkpoint['model']
        logging.info("Found 'model' in checkpoint")
    else:
        raise KeyError(f"No model state dict found in checkpoint. Keys found: {checkpoint.keys()}")
    
    logging.info(f"Trained state dict keys: {len(trained_state_dict.keys())}")
    
    # Log some key statistics before update
    missing_keys = set(full_state_dict.keys()) - set(trained_state_dict.keys())
    extra_keys = set(trained_state_dict.keys()) - set(full_state_dict.keys())
    logging.info(f"Keys missing in trained checkpoint: {len(missing_keys)}")
    logging.info(f"Extra keys in trained checkpoint: {len(extra_keys)}")
    
    # Update only the trained parameters
    updated_keys = 0
    for key in trained_state_dict:
        if key in full_state_dict:
            full_state_dict[key] = trained_state_dict[key]
            updated_keys += 1
    
    logging.info(f"Updated {updated_keys} keys in full state dict")
    
    # Create new checkpoint with full state dict
    new_checkpoint = {
        'model_state_dict': full_state_dict,
        'config': checkpoint.get('config', None),
        'args': checkpoint.get('args', None)
    }
    
    if save_path:
        torch.save(new_checkpoint, save_path)
        logging.info(f"Saved converted checkpoint to {save_path}")
        
        # Verify saved checkpoint
        verify = torch.load(save_path, map_location='cpu')
        logging.info(f"Verified saved checkpoint has {len(verify['model_state_dict'].keys())} keys")
    
    return new_checkpoint