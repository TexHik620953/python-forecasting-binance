import pandas as pd

# SMA
def add_sma(df):
    # Простые скользящие средние (SMA)
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_21'] = df['close'].rolling(window=21).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Экспоненциальные скользящие средние (EMA)
    df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Нормализованные отклонения от средних
    df['feature_sma_7_norm'] = (df['close'] - df['sma_7']) / df['close']
    df['feature_sma_21_norm'] = (df['close'] - df['sma_21']) / df['close']
    df['feature_sma_50_norm'] = (df['close'] - df['sma_50']) / df['close']

    df['feature_ema_7_norm'] = (df['close'] - df['ema_7']) / df['close']
    df['feature_ema_21_norm'] = (df['close'] - df['ema_21']) / df['close']
    df['feature_ema_50_norm'] = (df['close'] - df['ema_50']) / df['close']
# RSI
def add_rsi(df):
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


    df['rsi_7'] = calculate_rsi(df['close'], window=7)
    df['rsi_14'] = calculate_rsi(df['close'], window=14)
    df['rsi_21'] = calculate_rsi(df['close'], window=21)

    # Нормализованный RSI (0-1)
    df['feature_rsi_7_norm'] = df['rsi_7'] / 100
    df['feature_rsi_14_norm'] = df['rsi_14'] / 100
    df['feature_rsi_21_norm'] = df['rsi_21'] / 100
# MACD
def add_macd(df):
    # Вычисляем EMA 12 и EMA 26
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()

    # MACD = EMA(12) - EMA(26)
    macd = ema_12 - ema_26

    # Сигнальная линия (EMA 9 от MACD)
    signal = macd.ewm(span=9, adjust=False).mean()

    # Гистограмма MACD
    histogram = macd - signal

    # Сохраняем в DataFrame
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = histogram

    # Нормализованные фичи (относительно цены)
    df['feature_macd_norm'] = df['macd'] / df['close']
    df['feature_macd_signal_norm'] = df['macd_signal'] / df['close']
    df['feature_macd_hist_norm'] = df['macd_hist'] / df['close']


    # Удаляем промежуточные столбцы (если не нужны)

