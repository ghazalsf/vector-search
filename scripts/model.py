import clip
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_model():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess
