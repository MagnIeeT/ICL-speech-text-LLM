import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_checkpoint(path, map_location=None):
    """Load a checkpoint from disk."""
    logger.info("Loading checkpoint from %s", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    if "model_state_dict" in checkpoint:
        num_params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
        logger.info("Loaded checkpoint with %d parameters", num_params)

    return checkpoint
