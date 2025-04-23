import gymnasium as gym
import gym_trading_env
import os
import pandas as pd
import numpy as np
from neural.predictor import *
import torch
import matplotlib.pyplot as plt
from dataset.builder import add_sma, add_rsi, add_macd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictor = SimpleClassifier("market-transformer").to(device)
predictor.load()

df = pd.read_parquet("/mnt/y/DATASETS/Binance/ETHUSDT.parquet", engine='pyarrow')

add_sma(df)
add_rsi(df)
add_macd(df)
df.dropna(inplace=True)

data = df.to_numpy(dtype=np.float32)
env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0,#0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0#0.0003/100/60, # 0.0003% per timestep (one timestep = 1h here)
)

active_positon = None

def strategy(window, info, observation):
    global active_positon
    if info['idx'] % 30 == 0:
        with torch.no_grad():
            window = (window - window.mean(dim=1)) / (window.std(dim=1) + 1e-6)
            pred = predictor(window).cpu()[0]

        if pred.isnan().any().item():
            active_positon = None
            return

        arx = pred.argmax().item()

        if arx == 1:
            active_positon = 1
        else:
            confidence = pred[arx].item() / (abs(pred[0].item()) + abs(pred[2].item()))
            if confidence < 0.7:
                active_positon = 1
                return
            if arx == 0:
                active_positon = 0
            elif arx == 2:
                active_positon = 2




done, truncated = False, False
obs, info = env.reset()
portfolio_history = []
while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    position_index = env.action_space.sample()
    window = data[info['idx']:info['idx']+512].reshape(1,512,-1)
    window = torch.from_numpy(window).float().to(device)

    position = 1
    strategy(window, info, obs)

    if active_positon is not None:
        position = active_positon

    obs, reward, done, truncated, info = env.step(position)

    portfolio_history.append(info['portfolio_valuation'])
    print(info['portfolio_valuation'])


plt.plot(portfolio_history)
plt.show()