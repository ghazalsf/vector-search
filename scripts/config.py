import torch
DATA_PATH = "/home/divar/Documents/divar-projects/vector-search/data/images.parquet"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERY_IMAGE_PATH = '/home/divar/Documents/divar-projects/vector-search/data/2023-12-15/image/AY_cVWT6/AY_cVWT6.jpg'
