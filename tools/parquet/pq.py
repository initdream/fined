import pandas as pd
df = pd.read_parquet('filename.parquet')
df.to_csv('filename.csv')
