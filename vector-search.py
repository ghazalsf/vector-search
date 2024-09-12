import pandas as pd
import nmslib
import numpy as np
import torch
import clip
from PIL import Image
import gradio as gr

class VectorSearcher:
    def __init__(self, data_path, model_name="openai/clip-vit-base-patch32"):
        self.data_path = data_path
        self.model_name = model_name
        self.image_df = None
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def read_data(self):
        self.image_df = pd.read_parquet(self.data_path)
        print("Done - read data!")
        return self.image_df

    def initialize_model(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("Done - initialize model")
    
    def create_embedding_index(self, embedding_col='embedding', method='hnsw', space='cosinesimil', post=2):
        if self.image_df is None:
            raise ValueError("Data has not been read. Call `read_data` method first.")
        
        data_embeddings = np.vstack(self.image_df[embedding_col].values)
        index = nmslib.init(method=method, space=space)
        index.addDataPointBatch(data_embeddings)
        index.createIndex({'post': post}, print_progress=True)
        print("Done - create_embedding_index")
        return index

    def generate_image_embedding(self, image: Image.Image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image).cpu().numpy()[0]
        return embedding

    def find_nearest_images(self, query_image: Image.Image, index, k=3):
        query_embedding = self.generate_image_embedding(query_image)
        ids, distances = index.knnQuery(query_embedding, k=k)
        similar_images = self.image_df.iloc[ids]
        return similar_images.head(k), ids[:k], distances[:k]
    
    def search_and_display(self, query_image):
        similar_images, ids, distances = self.find_nearest_images(query_image, self.index, k=3)
        match_paths = self.image_df.iloc[ids]['image_path'].tolist()
        images = [Image.open(path) for path in match_paths]
        distances_list = list(map(float, distances))
        return images[0], distances_list[0]

    def gradio_interface(self):
        interface = gr.Interface(
            fn=self.search_and_display,
            inputs=gr.Image(type="pil", label="Upload Image"),
            outputs=[
                gr.Image(type="pil", label="Closest Match"),
                gr.Textbox(label="Distance")
            ],
            title="Image Search using CLIP and nmslib"
        )
        interface.launch()

if __name__ == "__main__":
    handler = VectorSearcher("/home/divar/Documents/divar-projects/vector-search/data/images.parquet")
    
    handler.read_data()
    
    handler.initialize_model()
    
    handler.index = handler.create_embedding_index()
    
    query_image_path = '/home/divar/Documents/divar-projects/vector-search/data/2023-12-15/image/AY_cVWT6/AY_cVWT6.jpg'
    query_image = Image.open(query_image_path)
    similar_images, ids, distances = handler.find_nearest_images(query_image, handler.index, k=3)
    print(f'IDs: {ids}, Distances: {distances}')
    
    # Launch Gradio Interface
    handler.gradio_interface()
