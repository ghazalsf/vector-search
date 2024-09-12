import pandas as pd
import logging

def read_data(data_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Reading data from {data_path}")
    image_df = pd.read_parquet(data_path)
    logger.info("Done - read data!")
    return image_df
