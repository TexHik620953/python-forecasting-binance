import os
import pandas as pd
import numpy as np

base_path = "/mnt/y/DATASETS/Binance"
pairs = [
"BTCUSDT.parquet",
"ETHUSDT.parquet",
"SOLUSDT.parquet",
"XRPUSDT.parquet",
]

dataframes = []
# OPEN HIGH LOW CLOSE VOLUME QUOTE_VOLUME NUMBER_OF TRADES TAKER_VOLUME TAKER_QUOTE_VOLUME

for pair in pairs:
    path = os.path.join(base_path, pair)
    df = pd.read_parquet(path, engine='pyarrow')
    dataframes.append(df)


common_index = dataframes[0].index
for df in dataframes[1:]:
    common_index = common_index.intersection(df.index)

# 2. Обрежем все датафреймы по общим временным меткам
aligned_dfs = []
for df in dataframes:
    aligned_df = df.loc[common_index]
    aligned_dfs.append(aligned_df)

# 3. Проверим, что все датафреймы теперь одинаковой длины
lengths = [len(df) for df in aligned_dfs]
assert len(set(lengths)) == 1, "Датафреймы имеют разную длину после выравнивания"

data = np.vstack([np.array([d.to_numpy()]) for d in aligned_dfs])
np.save("./data.npy", data)