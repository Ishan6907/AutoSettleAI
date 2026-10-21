import pandas as pd
from huggingface_hub import hf_hub_download

try:
    path = hf_hub_download(repo_id="TornikeO/CarAccidents", filename="data/train-00000-of-00003.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    print("Columns:", df.columns.tolist())
    print("Rows in shard 0:", len(df))
    print("Sample row:", type(df.iloc[0]['image']))
except Exception as e:
    print("Error:", e)
