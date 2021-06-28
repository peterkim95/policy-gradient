
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

if sys.argv[1] == 'ec2':
    from pyvirtualdisplay import Display
    dis = Display(visible=0, size=(1000, 1000))
    dis.start()

env = gym.make("Pong-v0")

NOOP = 0
UP_ACTION = 2
DOWN_ACTION = 3

_ = env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

transform = T.Compose([T.ToPILImage(),
                       T.Grayscale(),
                       T.Resize(80),
                       T.ToTensor()])

def preprocess(x):
    #     screen = env.render(mode='rgb_array') # (H,W,C) = (216,160,3)
    x = x[35:195, :] # cut out the score and border
    return torch.flatten(transform(x))


class Agent(nn.Module):
    def __init__(self, env, input_dim, output_dim):
        super(Agent, self).__init__()
        self.env = env

        self.h = nn.Linear(input_dim, 200)
        self.out = nn.Linear(200, output_dim)

    def forward(self, x):
        x = F.relu(self.h(x))
        return self.out(x)

    def set_weights(self, weights):
        index = 0
        for parameter in self.parameters():
            parameter.data.copy_(torch.from_numpy(weights[index:index+parameter.numel()].reshape(parameter.shape)))
            index += parameter.numel()

    def get_weights_dim(self):
        dim = 0
        for parameter in self.parameters():
            dim += parameter.numel()
        return dim

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0
        state = self.env.reset()
        prev_x = preprocess(state)
        done = False
        while not done:
            self.env.render()
            cur_x = preprocess(state)
            x = cur_x - prev_x
            prev_x = cur_x

            logits = self.forward(x.unsqueeze(0))
            sampler = Categorical(logits=logits)
            action = sampler.sample()

            state, reward, done, info = self.env.step(action.item()+1)
            episode_return += reward

        return episode_return


agent = Agent(env,6400,3)
running_reward = None

pop_size = 50
elite_frac = 0.2
n_elite = int(pop_size * elite_frac)
mean = 0.0
std = 1.0
best_weight = std * np.random.randn(agent.get_weights_dim())
epoch = 1
while True:
    weights_pop = [best_weight + std * np.random.randn(agent.get_weights_dim()) for _ in range(pop_size)]
    rewards = np.array([agent.evaluate(weights) for weights in weights_pop])
    print(f'epoch {epoch} pop rewards: {rewards}')
    elite_idxs = rewards.argsort()[-n_elite:]
    elite_weights = [weights_pop[i] for i in elite_idxs]

    best_weight = np.array(elite_weights).mean(axis=0)

    reward_sum = agent.evaluate(best_weight)
    running_reward = 0.99 * running_reward + 0.01 * reward_sum if running_reward is not None else reward_sum

    writer.add_scalar('Running Reward', running_reward, epoch)
    epoch += 1