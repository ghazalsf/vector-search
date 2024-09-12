from PIL import Image
import torch
import logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_image_embedding(model, preprocess, image: Image.Image):
    logger = logging.getLogger(__name__)
    logger.info("Generating image embedding")
    image = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy()[0]
    logger.info("Generated image embedding successfully")
    return embedding

class VectorSearcher:
    def __init__(self, image_df, model, preprocess):
        self.image_df = image_df
        self.model = model
        self.preprocess = preprocess
        self.logger = logging.getLogger(__name__)

    def find_nearest_images(self, query_image: Image.Image, index, k=3):
        self.logger.info("Finding nearest images")
        query_embedding = generate_image_embedding(self.model, self.preprocess, query_image)
        ids, distances = index.knnQuery(query_embedding, k=k)
        similar_images = self.image_df.iloc[ids]
        self.logger.info(f"Found {k} nearest images")
        return similar_images.head(k), ids[:k], distances[:k]

    def search_and_display(self, query_image, index):
        self.logger.info("Searching and displaying nearest image")
        similar_images, ids, distances = self.find_nearest_images(query_image, index, k=3)
        match_paths = self.image_df.iloc[ids]['image_path'].tolist()
        images = [Image.open(path) for path in match_paths]
        distances_list = list(map(float, distances))
        self.logger.info("Search and display completed")
        return images[0], distances_list[0]
