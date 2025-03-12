import gymnasium as gym
import gym_trading_env
import numpy as np
import pandas as pd


class TradingEnvBatched:
    def __init__(self, dataset, columns, target_index, split_number = 250, window_size = 32):
        self.window_size = window_size
        self.target_index = target_index
        self.split_number = split_number
        self.columns = columns
        self.dataset_chunks = np.array_split(dataset, split_number)

        self.infos = [None for _ in range(split_number)]
        self.envs = []
        self.position = 0
        for i in range(len(self.dataset_chunks)):
            df = pd.DataFrame(self.dataset_chunks[i][window_size:,self.target_index], columns=columns, dtype='float64')
            self.envs.append(gym.make("TradingEnv",
                       df=df,  # Your dataset with your custom features
                       positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                       trading_fees=0.005,
                       borrow_interest_rate=0,  # 0.0003% per timestep (one timestep = 1h here)
                       dynamic_feature_functions=[]
                       ))

    def reset(self):
        self.infos = [None for _ in range(self.split_number)]
        self.envs = []
        self.position = 0
        for i in range(len(self.dataset_chunks)):
            df = pd.DataFrame(self.dataset_chunks[i][self.window_size:, self.target_index], columns=self.columns, dtype='float64')
            self.envs.append(gym.make("TradingEnv",
                                      df=df,  # Your dataset with your custom features
                                      positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                                      trading_fees=0.001,
                                      borrow_interest_rate=0.0003 / 100,
                                      # 0.0003% per timestep (one timestep = 1h here)
                                      dynamic_feature_functions=[]
                                      ))

    def __norm(self, obs):
        std = np.std(obs, keepdims=True, axis=(1))
        mean = np.mean(obs, keepdims=True, axis=(1))
        return (obs - mean) / (std + 1e-5)

    def start(self):
        window_states = []
        latents = []
        for i in range(len(self.envs)):
            obs, info = self.envs[i].reset()
            self.infos[i] = info
            state = self.dataset_chunks[i][info['idx']+1:self.window_size + info['idx']+1]

            window = state[:, :, :-1]

            time_onehot = np.eye(60)[int(state[-1, self.target_index, -1])]
            position = np.eye(3)[info['position_index']]

            latents.append(np.concatenate([time_onehot, position]))
            window_states.append(window)

        window_states = np.array(window_states)
        window_states = window_states.reshape(window_states.shape[0], self.window_size, -1)
        window_states = self.__norm(window_states)
        window_states = window_states.swapaxes(1, 2)

        latents = np.array(latents)
        latents = np.array(latents, dtype='float64')

        return [window_states, latents]

    def step(self, actions):
        self.position += 1

        if self.position > self.dataset_chunks[0].shape[0] - self.window_size*2:
            self.reset()

        window_states = []
        latents = []
        rewards = []
        for i in range(len(self.envs)):
            observation, reward, done, truncated, info = self.envs[i].step(np.argmax(actions[i]))
            self.infos[i] = info
            rewards.append(reward)
            state = self.dataset_chunks[i][info['idx'] + 1:self.window_size + info['idx'] + 1]

            window = state[:, :, :-1]

            time_onehot = np.eye(60)[int(state[-1, self.target_index, -1])]
            position = np.eye(3)[info['position_index']]

            latents.append(np.concatenate([time_onehot, position]))
            window_states.append(window)

        window_states = np.array(window_states)
        window_states = window_states.reshape(window_states.shape[0], self.window_size, -1)
        window_states = self.__norm(window_states)
        window_states = window_states.swapaxes(1, 2)

        latents = np.array(latents)
        latents = np.array(latents, dtype='float64')

        rewards = np.array(rewards, dtype='float64')

        return [window_states, latents], actions, np.array(rewards)
