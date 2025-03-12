import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

from neural.pytorch_a2c.encoder import Encoder
from neural.pytorch_a2c.actor import Actor
from neural.pytorch_a2c.critic import Critic

class Brain(nn.Module):
    def __init__(self, name, enable_tb=True, critic_learning_rate=5e-4, actor_learning_rate=1e-4,
                 epsilon=0.1, epsilon_decay=0.9995, actions_amount=3, window_latent_size=25, window_size=32, features_size=3,
                 internal_features_size=128):
        super(Brain, self).__init__()
        self.window_size = window_size
        self.name = name
        self.enable_tb = enable_tb
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.actions_amount = actions_amount
        self.window_latent_size = window_latent_size
        self.features_size = features_size
        self.internal_features_size = internal_features_size

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Launching on GPU")
        else:
            self.device = torch.device("cpu")
            print("Launching on CPU")

        self.encoder = Encoder(self.window_size, self.window_latent_size, self.features_size, self.internal_features_size).to(self.device)
        self.actor = Actor(self.encoder, self.internal_features_size, self.actions_amount).to(self.device)

        self.critic = Critic(self.encoder, self.internal_features_size).to(self.device)


        self.target_critic = Critic(
            Encoder(self.window_size,
                    self.window_latent_size,
                    self.features_size,
                    self.internal_features_size).to(self.device),
            self.internal_features_size
        ).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=0.001)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate, weight_decay=0.001)

        if self.enable_tb:
            self.logdir = f"./tensorboard/actor"
            self.tensorboard = SummaryWriter(self.logdir)
            self.step = 0

        self.profile()
        try:
            self.load_model()
        except Exception as e:
            print(f"[{self.name}]Failed to load model, creating new one: {e}")


    def profile(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            state = (
                torch.empty(512, self.window_latent_size, self.window_size).normal_(mean=1,std=1).to(self.device),
                torch.empty(512, self.features_size).normal_(mean=1, std=1).to(self.device)
            )
            with record_function("actor"):
                self.actor(state)
            with record_function("critic"):
                self.critic(state)
        print(prof.key_averages().table(row_limit=5))


    def get_action(self, observations):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < 0.01:
            self.epsilon = 0.01

        observations = (
            torch.tensor(observations[0], dtype=torch.float32).to(self.device),
            torch.tensor(observations[1], dtype=torch.float32).to(self.device)
        )

        act_prob = self.actor(observations).detach().cpu().numpy()

        for i in range(observations[0].shape[0]):
            if random.random() < self.epsilon:
                act_prob[i] = np.random.uniform(0, 1, size=self.actions_amount)
                act_prob[i] /= np.sum(act_prob[i])
        return act_prob

    def train_step(self, trajectories, critic_repetitons=10, actor_repetitions=10):
        actions = torch.tensor(np.vstack(trajectories[1]), dtype=torch.float32).to(self.device)
        _rewards = torch.tensor(np.expand_dims(trajectories[2], axis=1), dtype=torch.float32).to(self.device)

        states = [torch.tensor(np.array(x,dtype='float32'), dtype=torch.float32).to(self.device) for x in list(map(list, zip(*[s for s in trajectories[0]])))]
        new_states = [torch.tensor(np.array(x,dtype='float32'), dtype=torch.float32).to(self.device) for x in list(map(list, zip(*[s for s in trajectories[3]])))]

        #rewards = (_rewards - torch.mean(_rewards)) / torch.std(_rewards)
        rewards = _rewards

        # Critic steps
        advantages = None
        critic_loss = None
        for _ in range(critic_repetitons+1):
            self.critic_optimizer.zero_grad()

            advantages = (rewards + 0.98 * self.target_critic(new_states) - self.critic(states))
            critic_loss = torch.mean(torch.square(advantages))

            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        tau = 0.005  # Коэффициент обновления
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        critic_loss = critic_loss.detach().cpu().numpy()
        advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        advantages = advantages.detach()


        actions_prob = None
        entropy_loss = None
        actor_loss = None
        for _ in range(actor_repetitions+1):
            self.actor_optimizer.zero_grad()

            actions_prob = self.actor(states)
            # Entropy
            entropy_loss = torch.mean(torch.sum(actions_prob * torch.log(actions_prob + 1e-10), dim=1))  # Энтропия, maximize this

            action_masks = torch.nn.functional.one_hot(torch.argmax(actions, dim=1), num_classes=self.actions_amount)
            new_probs = torch.sum(action_masks * actions_prob, dim=1, keepdim=True)
            old_probs = torch.sum(action_masks * actions, dim=1, keepdim=True)

            ratio = torch.exp(torch.log(new_probs + 1e-10) - torch.log(old_probs + 1e-10))

            actor_loss = -torch.mean(ratio * advantages) + 0.01 * torch.mean(entropy_loss)

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

        actor_loss = actor_loss.detach().cpu().numpy()
        entropy_loss = entropy_loss.detach().cpu().numpy()
        actions_prob = actions_prob.cpu().detach()

        if self.enable_tb:
            actions_idx_prob = torch.argmax(actions_prob, dim=1)
            self.tensorboard.add_histogram("actions", actions_prob, self.step)
            self.tensorboard.add_histogram("action_idx", actions_idx_prob, self.step)
            self.tensorboard.add_scalar("entropy_loss", entropy_loss, self.step)
            self.tensorboard.add_scalar("epsilon", self.epsilon, self.step)
            self.tensorboard.add_scalar("loss/actor", actor_loss, self.step)
            self.tensorboard.add_scalar("loss/critic", critic_loss, self.step)
            self.tensorboard.add_scalar("rewards", torch.mean(_rewards).cpu(), self.step)
            if self.step % 10 == 0:
                for name, param in self.actor.named_parameters():
                    self.tensorboard.add_histogram(f"actor_grad/{name}", param.grad, self.step)
                for name, param in self.critic.named_parameters():
                    self.tensorboard.add_histogram(f"critic_grad/{name}", param.grad, self.step)

        if self.step % 100 == 0:
            self.save_model()
        self.step += 1
        return advantages.cpu().numpy().squeeze()

    def save_model(self):
        try:
            os.makedirs(f"./checkpoints/{self.name}", exist_ok=True)

            torch.save({
                'encoder': self.encoder.state_dict(),
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'step': self.step,
                'epsilon': self.epsilon,
            }, f'./checkpoints/{self.name}/actor.weights.pth')
        except Exception as e:
            print(f"[!!!]Failed to save model {self.name}: {e}")

    def load_model(self):
        data = torch.load(f'./checkpoints/{self.name}/actor.weights.pth')
        self.actor.load_state_dict(data['actor'])

        self.critic.load_state_dict(data['critic'])
        self.target_critic.load_state_dict(data['critic'])

        self.encoder.load_state_dict(data['encoder'])

        self.actor_optimizer.load_state_dict(data['actor_optimizer'])
        self.critic_optimizer.load_state_dict(data['critic_optimizer'])

        self.step = data['step']
        self.epsilon = data['epsilon']