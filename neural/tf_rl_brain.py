import random
import tensorflow as tf
import os
import numpy as np

def dense_residual_block(x, units):
    shortcut = x

    x = tf.keras.layers.Dense(units, activation=None, kernel_initializer='he_uniform', bias_initializer='he_uniform')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Dense(units, activation=None, kernel_initializer='he_uniform', bias_initializer='he_uniform')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Если размерность входа и выхода не совпадают, добавляем проекцию
    if shortcut.shape[-1] != units:
        shortcut = tf.keras.layers.Dense(units, activation=None, kernel_initializer='he_uniform', bias_initializer='he_uniform')(shortcut)

    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    return x

def conv1d_residual(input, filters, kernel_size=3, pool=True):
    residual = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same', kernel_initializer='he_uniform', bias_initializer='he_uniform')(input)

    l = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_uniform', bias_initializer='he_uniform')(input)
    l = tf.keras.layers.LeakyReLU(0.2)(l)

    l = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_uniform', bias_initializer='he_uniform')(l)
    l = tf.keras.layers.LeakyReLU(0.2)(l)

    if pool:
        l = tf.keras.layers.AveragePooling1D(pool_size=2)(residual + l)
    else:
        l = tf.keras.layers.Add()([residual, l])

    o = tf.keras.layers.LayerNormalization()(l)
    return o

def build_history_handler_block(window_size, latent_size):
    features_input = tf.keras.layers.Input(shape=(window_size, latent_size,))
    l = features_input

    l = conv1d_residual(l, 32, kernel_size=5, pool=True)
    l = conv1d_residual(l, 48, kernel_size=5, pool=True)
    l = conv1d_residual(l, 64, kernel_size=3, pool=True)
    l = conv1d_residual(l, 64, kernel_size=3, pool=False)
    l = conv1d_residual(l, 64, kernel_size=3, pool=False)

    l = tf.keras.layers.GlobalMaxPooling1D()(l)
    output = tf.keras.layers.LayerNormalization()(l)
    return features_input, output

class Brain:
    def __init__(self,
                 name,
                 enable_tb=True,
                 encoder_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 actor_learning_rate=3e-4,
                 epsilon=0.1,
                 epsilon_decay=0.9995,
                 actions_amount=3,
                 window_latent_size=25,
                 window_size=32,
                 features_size = 3,
                 internal_features_size=64,
                 ):
        self.window_size = window_size
        self.name = name
        self.enable_tb =enable_tb
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.actions_amount = actions_amount
        self.window_latent_size = window_latent_size
        self.features_size = features_size
        self.internal_features_size = internal_features_size

        self.encoder = self.__build_encoder()
        self.critic = self.__build_critic()
        self.actor = self.__build_actor()


        self.encoder_optimizer = tf.optimizers.Adam(learning_rate=encoder_learning_rate)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=critic_learning_rate)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=actor_learning_rate)

        #plot_model(self.critic, to_file='./tmp/critic.png', show_shapes=True, show_layer_names=True, show_trainable=True,show_dtype=True)
        #plot_model(self.actor, to_file='./tmp/actor.png', show_shapes=True, show_layer_names=True, show_trainable=True,show_dtype=True)
        #plot_model(self.encoder, to_file='./tmp/encoder.png', show_shapes=True, show_layer_names=True, show_trainable=True,show_dtype=True)
        if self.enable_tb:
            self.logdir = f"./tensorboard/actor"
            self.tensorboard = tf.summary.create_file_writer(self.logdir, name=name)
            self.step = 0

            @tf.function
            def trace_model(model, input_data):
                return model(input_data)

            #Trace encoder
            tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=self.logdir)
            i = [tf.random.normal([100, self.window_size, self.window_latent_size]), tf.random.normal([100, self.features_size])]
            _ = trace_model(self.encoder, i)
            with self.tensorboard.as_default():
                tf.summary.trace_export(
                    name="encoder",
                    step=0,
                )
            tf.summary.trace_off()

            #Trace actor
            tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=self.logdir)
            i = tf.random.normal([100, self.internal_features_size])
            _ = trace_model(self.actor, i)
            with self.tensorboard.as_default():
                tf.summary.trace_export(
                    name="actor",
                    step=0,
                )
            tf.summary.trace_off()

            #Trace critic
            tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=self.logdir)
            i = tf.random.normal([100, self.internal_features_size])
            _ = trace_model(self.critic, i)
            with self.tensorboard.as_default():
                tf.summary.trace_export(
                    name="critic",
                    step=0,
                )
            tf.summary.trace_off()

        try:
            self.load_model()
        except Exception as e:
            print(f"[{self.name}]Failed to load model, creating new one: {e}")
    def get_action(self, observations):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < 0.01:
            self.epsilon = 0.01
        act_prob = self.actor(self.encoder(observations)).numpy()

        for i in range(observations[0].shape[0]):
            if random.random() < self.epsilon:
                act_prob[i] = np.random.uniform(0, 1, size=self.actions_amount)
                act_prob[i] /= np.sum(act_prob[i])
        return act_prob

    def __build_encoder(self):
        window_input, window_features = build_history_handler_block(self.window_size, self.window_latent_size)
        features = tf.keras.layers.Input(shape=(self.features_size,))

        common = tf.keras.layers.Concatenate()([window_features,features])

        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)
        common = tf.keras.layers.Dropout(0.1)(common)

        encoded_features = dense_residual_block(common, self.internal_features_size)

        return tf.keras.Model(inputs=[window_input,features], outputs=encoded_features)
    def __build_actor(self):
        encoded_features = tf.keras.layers.Input(shape=(self.internal_features_size,))
        common = encoded_features

        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)

        common = tf.keras.layers.Dropout(0.1)(common)
        suggested_action = tf.keras.layers.Dense(self.actions_amount, use_bias=False, activation="softmax")(common)

        return tf.keras.Model(inputs=encoded_features, outputs=suggested_action)

    def __build_critic(self):
        encoded_features = tf.keras.layers.Input(shape=(self.internal_features_size,))
        common = encoded_features

        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)
        common = dense_residual_block(common, 64)

        common = tf.keras.layers.Dropout(0.1)(common)
        val = tf.keras.layers.Dense(1, use_bias=False, activation=None)(common)

        return tf.keras.Model(inputs=encoded_features, outputs=val)


    def train_step(self, trajectories, repetiions = 10):
        # Best is A2C + PPO
        actions = np.array(trajectories[1], dtype='float32')
        _rewards = np.array(np.expand_dims(trajectories[2], axis=1), dtype='float32')

        states = [np.vstack(np.array([x]), dtype='float32') for x in
                  list(map(list, zip(*[s for s in trajectories[0]])))]
        new_states = [np.vstack(np.array([x]), dtype='float32') for x in
                      list(map(list, zip(*[s for s in trajectories[3]])))]


        rewards = _rewards
        #rewards = (_rewards - np.mean(_rewards)) / np.std(_rewards)

        for _ in range(repetiions):
            with tf.GradientTape() as critic_tape, tf.GradientTape() as actor_tape, tf.GradientTape() as encoder_tape:
                old_encoded = self.encoder(states)
                new_encoded = self.encoder(new_states)

                values = self.critic(old_encoded)
                next_values = self.critic(new_encoded)

                raw_advantages = (rewards + 0.9 * next_values - values)

                critic_loss = tf.reduce_mean(tf.square(raw_advantages)) #Minimize this

                critic_grads, _ = tf.clip_by_global_norm(
                    critic_tape.gradient(critic_loss, self.critic.trainable_variables), 0.5)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                advantages = (raw_advantages - tf.reduce_mean(raw_advantages)) / tf.math.reduce_std(raw_advantages)
                actions_prob = self.actor(old_encoded)
                # PPO
                ppo_epsilon = 0.2

                action_masks  = tf.one_hot(np.argmax(actions, axis=1), self.actions_amount)
                new_probs = tf.reduce_sum(action_masks * actions_prob, axis=1, keepdims=True)
                old_probs = tf.reduce_sum(action_masks * actions, axis=1, keepdims=True)

                log_new_probs = tf.math.log(new_probs + 1e-10)
                log_old_probs = tf.math.log(old_probs + 1e-10)
                ratio = tf.exp(log_new_probs - log_old_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - ppo_epsilon, 1 + ppo_epsilon)

                #Entropy
                entropy = -tf.reduce_mean(tf.reduce_sum(actions_prob * tf.math.log(actions_prob + 1e-10), axis=1))  # Энтропия, maximize this

                #L2 Actions loss
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages)) - 0.02 * tf.reduce_mean(entropy)

                actor_grads, _ = tf.clip_by_global_norm(
                    actor_tape.gradient(actor_loss, self.actor.trainable_variables), 0.5)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                total_loss = actor_loss + critic_loss

                encoder_grads, _ = tf.clip_by_global_norm(
                    encoder_tape.gradient(total_loss, self.encoder.trainable_variables), 0.5)
                self.encoder_optimizer.apply_gradients(zip(encoder_grads, self.encoder.trainable_variables))

        if self.enable_tb:
            with self.tensorboard.as_default():
                tf.summary.histogram("test/rewards", rewards, self.step)
                tf.summary.histogram("test/advantages", advantages, self.step)
                tf.summary.histogram("test/ratio", ratio, self.step)
                tf.summary.histogram("test/values", values, self.step)
                tf.summary.histogram("test/next_values", next_values, self.step)
                tf.summary.histogram("test/next_diff_values", next_values - values, self.step)
                actions_idx_prob = np.argmax(actions_prob, axis=1)
                tf.summary.histogram("actions", actions_prob, self.step)
                tf.summary.histogram("critics", values, self.step)
                tf.summary.histogram("action_idx", actions_idx_prob, self.step)
                tf.summary.scalar("actor_entropy", entropy, self.step)
                tf.summary.scalar("epsilon", self.epsilon, self.step)
                tf.summary.scalar("loss/actor", actor_loss, self.step)
                tf.summary.scalar("loss/critic", critic_loss, self.step)
                tf.summary.scalar("rewards", np.mean(_rewards), self.step)

        if self.step % 100 ==0:
            self.save_model()
        self.step += 1
        return np.abs(np.squeeze(raw_advantages.numpy())) + np.abs(np.squeeze(ratio.numpy()) - 1)

    def save_model(self):
        try:
            os.makedirs(f"./checkpoints/{self.name}", exist_ok=True)

            self.actor.save_weights(f'./checkpoints/{self.name}/actor.weights.h5')
            self.critic.save_weights(f'./checkpoints/{self.name}/critic.weights.h5')
            self.encoder.save_weights(f'./checkpoints/{self.name}/encoder.weights.h5')
            #Save step and optimizer
            np.save(f"./checkpoints/{self.name}/general.npy", {
                "step": self.step,
                "epsilon": self.epsilon
            }, allow_pickle=True)
        except Exception as e:
            print(f"[!!!]Failed to save model {self.name}: {e}")

    def load_model(self):
        self.actor.save_weights(f'./checkpoints/{self.name}/actor.weights.h5')
        self.critic.save_weights(f'./checkpoints/{self.name}/critic.weights.h5')
        self.encoder.save_weights(f'./checkpoints/{self.name}/encoder.weights.h5')

        data = np.load(f"./checkpoints/{self.name}/general.npy", allow_pickle=True).item()
        self.step = data["step"]
        self.epsilon = data["epsilon"]
