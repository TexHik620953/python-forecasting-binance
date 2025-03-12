import random
from dataset.dataset_raw import Dataset
from neural.tf_rl_brain import Brain
import tensorflow as tf
from neural.replay_buffer import ReplayBuffer

class HyperConfig:
    def __init__(self,
                 seed=123,
                 encoder_learning_rate=7e-4,
                 critic_learning_rate=7e-4,
                 actor_learning_rate=7e-4,
                 window_size=32,
                 internal_features_size=128,
                 ):
        self.seed = seed
        self.encoder_learning_rate = encoder_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.window_size = window_size
        self.internal_features_size = internal_features_size


TARGET_PAIR = "ETHUSDT"
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
_dataset=Dataset(symbols=symbols, align_to=TARGET_PAIR)
_data = _dataset.build_dataset()

replay_buffer = ReplayBuffer(1000000)
replay_buffer.load()
def train(steps: int, config:HyperConfig):

    random.seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(config.seed)

    brain=Brain(name="goga", enable_tb=False, window_latent_size=25, features_size=63,
                window_size=config.window_size,
                encoder_learning_rate=config.encoder_learning_rate,
                critic_learning_rate=config.critic_learning_rate,
                actor_learning_rate=config.actor_learning_rate,
                internal_features_size=config.internal_features_size
                )
    loss = 0
    for i in range(steps):
        minibatch = replay_buffer.sample_random_minibatch(1000)
        _, loss = brain.train_step(minibatch, 20)

