import os
import random

import pandas as pd
import numpy as np
from builder import *

base_path = "/mnt/y/DATASETS/Binance"
pair = "ETHUSDT.parquet"

dataframes = []
# OPEN HIGH LOW CLOSE VOLUME QUOTE_VOLUME NUMBER_OF TRADES TAKER_VOLUME TAKER_QUOTE_VOLUME

path = os.path.join(base_path, pair)
df = pd.read_parquet(os.path.join(base_path, pair), engine='pyarrow')


df['taker_buy_base_asset_volume'] = df['taker_buy_base_asset_volume'] / (df['volume'] + 1e-6)
df['taker_buy_quote_asset_volume'] = df['taker_buy_quote_asset_volume'] / (df['quote_asset_volume'] + 1e-6)

add_sma(df)
add_rsi(df)
add_macd(df)
#drop_cols = []
#df.drop(columns=drop_cols, inplace=True, errors='ignore')
# Удаляем строки с NaN (из-за rolling/ewm)
df.dropna(inplace=True)

data = df.to_numpy(dtype=np.float64)


WINDOW_SIZE = 512
LOOKAHEAD = 30

indexes = list(range(len(data) - WINDOW_SIZE - LOOKAHEAD))
random.shuffle(indexes)

x = []
y = []

c = 0
total = len(indexes)

part_num = 0

for i in indexes:
    if c % 1000 == 0:
        print(f"Progress: {c}/{total} [{c/total*100:.2f}%]")
    c += 1
    _x = data[i:i + WINDOW_SIZE]
    _x = (_x - np.mean(_x, axis=0)) / np.std(_x, axis=0)
    x.append(_x)

    _y = (data[i+WINDOW_SIZE + LOOKAHEAD - 1][3] - data[i + WINDOW_SIZE -1][3]) / data[i + WINDOW_SIZE -1][3]
    y.append(_y)

    if len(x) > 20000:
        x = np.array(x)
        y = np.array(y)
        np.save(f"data{part_num}.npy", {
            "x": x,
            "y": y
        }, allow_pickle=True)
        x = []
        y = []
        part_num+=1


x = np.array(x)
y = np.array(y)

np.save(f"data{part_num}.npy", {
    "x": x,
    "y": y
}, allow_pickle=True)