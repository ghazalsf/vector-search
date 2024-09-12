import pandas as pd

def read_data(data_path):
    image_df = pd.read_parquet(data_path)
    print("Done - read data!")
    return image_df
