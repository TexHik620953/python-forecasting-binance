from dataset.dataset_raw import Dataset, LOAD_COLUMNS
from env.trading_env import TradingEnvBatched
from neural.prioritized_replay_buffer import PrioritizedReplayBuffer
from neural.pytorch_a2c.rl_brain import Brain
import numpy as np
from neural.replay_buffer import ReplayBuffer

TARGET_PAIR = "ETHUSDT"
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
dataset=Dataset(symbols=symbols, align_to=TARGET_PAIR)
data = dataset.build_dataset()

WINDOW_SIZE=24

MINIBATCH_SIZE = 1000

env = TradingEnvBatched(data, LOAD_COLUMNS, split_number = 1000, target_index=symbols.index(TARGET_PAIR), window_size=WINDOW_SIZE)
brain=Brain(name="goga", window_size=WINDOW_SIZE, window_latent_size=25, features_size=63)
replay_buffer = ReplayBuffer(27000)
#replay_buffer = PrioritizedReplayBuffer(27000)

state = env.start()
for k in range(2000):
    act = brain.get_action(state)
    new_state, act, rewards = env.step(act)
    for i in range(state[0].shape[0]):
        replay_buffer.add_replay((state[0][i], state[1][i]), act[i], rewards[i], (new_state[0][i], new_state[1][i]))
    state = new_state

    print(f"[{len(replay_buffer)}]Portfolio: {np.mean([x['portfolio_valuation'] for x in env.infos])}")
    if len(replay_buffer) > MINIBATCH_SIZE*2:
        print("Training")
        #minibatch, indices, _ = replay_buffer.sample_random_minibatch(0, MINIBATCH_SIZE)
        #priorities = brain.train_step(minibatch, 15, 15)
        #replay_buffer.update_priorities(0, indices, np.abs(priorities)+1e-5)

        minibatch = replay_buffer.sample_random_minibatch(MINIBATCH_SIZE)
        brain.train_step(minibatch, 40, 30)
    if k % 400 == 0:
        replay_buffer.save()
