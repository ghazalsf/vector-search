import clip
import torch
import logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_model():
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing model on device {DEVICE}")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    logger.info("Model initialized successfully")
    return model, preprocess
