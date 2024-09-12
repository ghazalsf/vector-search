import numpy as np
import nmslib
import logging

def create_embedding_index(image_df, embedding_col='embedding', method='hnsw', space='cosinesimil', post=2):
    logger = logging.getLogger(__name__)
    logger.info("Creating embedding index")
    data_embeddings = np.vstack(image_df[embedding_col].values)
    index = nmslib.init(method=method, space=space)
    index.addDataPointBatch(data_embeddings)
    index.createIndex({'post': post}, print_progress=True)
    logger.info("Done - create_embedding_index")
    return index
