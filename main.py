from scripts.preprocess import read_data
from scripts.model import initialize_model
from scripts.index import create_embedding_index
from scripts.vector_search import VectorSearcher
from scripts.gradio_interface import gradio_interface
from PIL import Image

DATA_PATH = "data/images.parquet"
QUERY_IMAGE_PATH = 'data/2023-12-15/image/AY_cVWT6/AY_cVWT6.jpg'

if __name__ == "__main__":
    image_df = read_data(DATA_PATH)
    model, preprocess = initialize_model()
    
    # Initialize the index
    index = create_embedding_index(image_df)
    vector_searcher = VectorSearcher(image_df, model, preprocess)
    
    # Test with a single query image
    query_image = Image.open(QUERY_IMAGE_PATH)
    similar_images, ids, distances = vector_searcher.find_nearest_images(query_image, index, k=3)
    print(f'IDs: {ids}, Distances: {distances}')

    # Launch the Gradio Interface
    gradio_interface(vector_searcher, index)
