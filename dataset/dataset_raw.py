import os
import pandas as pd
from glob import glob
import numpy as np
import pickle

columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
]

LOAD_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'feature_minute_onehot']
class Dataset:
    def __init__(self,
                 base_path = "./dataset/raw/data/futures/um/monthly/klines/",
                 cache_path = "./dataset/cached/",
                 symbols=None,
                 align_to ="ETHUSDT"
                 ):
        self.interval = "1m"
        self.target_pair = align_to

        if align_to not in symbols:
            raise "Target_pair must be in symbols"

        self.base_path = base_path
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)

        self.symbols = symbols

    def build_dataset(self):
        temp = self.__load_dict_data()
        data = []
        for sym in self.symbols:
            data.append(temp[sym])
        data = np.array(data).swapaxes(0,1) #(2675520, 11, 6)

        return data

    def __load_dict_data(self):
        data = self.__load_symbol_cache("GLOBAL")
        if data is not None:
            return data.item()

        data = dict()

        for symbol in self.symbols:
            data[symbol] = self.load_pair(symbol)
        target_pair = data[self.target_pair]

        for symbol in self.symbols:
            original = data[symbol]
            temp = original.reindex(target_pair.index)
            temp_filled = temp.ffill().bfill()
            data[symbol] = temp_filled
            if temp_filled.isna().any().any():
                raise f"Warning: {symbol} still has missing values after filling."

            np_arr = temp_filled[LOAD_COLUMNS].to_numpy(dtype="float64")
            data[symbol] = np_arr

        self.__save_symbol_cache("GLOBAL", data)
        return data

    def __save_symbol_cache(self, symbol, data):
        path = os.path.join(self.cache_path, symbol + ".npy")
        np.save(path, data, allow_pickle=True)

    def __load_symbol_cache(self, symbol):
        path = os.path.join(self.cache_path, symbol + ".npy")
        if os.path.exists(path):
            return np.load(path, allow_pickle=True)
        return None

    def load_pair(self, symbol):
        files = glob(os.path.join(self.base_path, symbol, self.interval, "*.csv"))
        dataframes = []
        for file in files:
            with open(file, 'r') as f:
                first_line = f.readline().strip()
            # Если первая строка содержит заголовок, пропускаем её
            if first_line == ",".join(columns):
                df = pd.read_csv(file, header=0)
            else:
                df = pd.read_csv(file, header=None)
            df.columns = columns
            dataframes.append(df)

        # Объединяем все DataFrame в один
        combined_df = pd.concat(dataframes, ignore_index=True)
        # Сортируем по времени
        combined_df = combined_df.sort_values(by='open_time')

        combined_df = combined_df.set_index('open_time')

        combined_df['feature_minute_onehot'] =np.array((combined_df.index/1000/60) % 60, dtype='int').tolist()

        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)


        return combined_df


