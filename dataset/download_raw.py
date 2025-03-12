import pandas as pd
import datetime
from binance_bulk_downloader.downloader import BinanceBulkDownloader

pairs = ["SOLUSDT"]
# Основная функция
def download_history():
    downloader = BinanceBulkDownloader(
        data_type="klines",
        asset="um",
        data_frequency="1m",
        timeperiod_per_file="monthly",
        destination_dir= "./raw",
        symbols=pairs,
    )
    downloader.run_download()

if __name__ == "__main__":
    download_history()