import gradio as gr
from .vector_search import VectorSearcher

def gradio_interface(vector_searcher, index):
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
