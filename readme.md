# Vector Search Project

This project demonstrates an image search application using OpenAI's CLIP model and nmslib for efficient similarity search. With Gradio, a user-friendly interface is designed to allow users to upload images and find similar ones from a dataset.

## Project Structure

```plaintext
vector_search_project/
├── data/
│   └── images.parquet       # Example dataset file
├── scripts/
│   ├── model.py             # Initialization of the CLIP model
│   ├── preprocess.py        # Preprocessing functionalities
│   ├── index.py             # Indexing functionalities using nmslib
│   ├── vector_search.py     # Vector search functionality and operations
│   └── interface.py         # Gradio interface for image search
├── main.py                  # Entry point of the application
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/vector_search_project.git
    cd vector_search_project
    ```

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare your data:**

    Ensure your dataset is available in the `data/` directory, and the path is correctly set in the scripts.

## Running the Application

1. **Run the main script:**

    ```sh
    python main.py
    ```

2. **Using the Gradio interface:**

    The Gradio web interface will be launched automatically, allowing you to upload images and find similar ones in the dataset.

## Project Description

### Scripts

- **model.py:** Contains the function for initializing the CLIP model.
- **preprocess.py:** Functionality for reading and preprocessing the dataset.
- **index.py:** Creation of the nmslib index for efficient similarity search.
- **vector_search.py:** Core vector search functionality, including generating image embeddings and finding nearest images.
- **interface.py:** Sets up the Gradio interface for the image search application.

### Data

- **data/images.parquet:** Example dataset file. Ensure your data is structured appropriately for reading and processing.

## Dependencies

All dependencies are listed in `requirements.txt`. The main libraries used include:

- `pandas`
- `nmslib`
- `numpy`
- `torch`
- `clip-by-openai`
- `Pillow`
- `gradio`

