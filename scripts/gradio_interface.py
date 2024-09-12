import gradio as gr
from .vector_search import VectorSearcher
import logging

def gradio_interface(vector_searcher, index):
    logger = logging.getLogger(__name__)
    logger.info("Launching Gradio interface")
    interface = gr.Interface(
        fn=lambda query_image: vector_searcher.search_and_display(query_image, index),
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[
            gr.Image(type="pil", label="Closest Match"),
            gr.Textbox(label="Distance")
        ],
        title="Image Search using CLIP and nmslib"
    )
    interface.launch()
