"""Reload a pretrained stage-1 model for node embedding extraction."""
import torch

from dataset import validate_contract
from models import CdfgNN


def load_encoder(path, *, device="cpu", preprocessing_fingerprint=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    validate_contract(checkpoint)
    if preprocessing_fingerprint is not None and checkpoint.get("preprocessing_fingerprint") != preprocessing_fingerprint:
        raise ValueError("Checkpoint preprocessing fingerprint does not match")
    model = CdfgNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model
