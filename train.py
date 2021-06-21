import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

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
    return transform(x)


class PolicyNetwork(nn.Module):
    '''
    (N,1,80,80) -> (N,3) probability distribution of 3 actions
    '''
    def __init__(self, h, w, outputs):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=8, stride=2)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(16)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w, 8, 2), 4, 2)
        convh = conv2d_size_out(conv2d_size_out(h, 8, 2), 4, 2)
        linear_input_size = 16 * convw * convh
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))


model = PolicyNetwork(80, 80, 3)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.99)


def select_action(x):
    x = x.unsqueeze(0)
    output = model(x)
    dist = Categorical(logits=output)
    # print(dist.probs)
    sampled_action = dist.sample()
    log_p = dist.log_prob(sampled_action)
    sampled_action = sampled_action.item()
    if sampled_action == 0:
        return NOOP, log_p
    elif sampled_action == 1:
        return UP_ACTION, log_p
    elif sampled_action == 2:
        return DOWN_ACTION, log_p


render = True
batch_size = 1
gamma = 0.99


def discount_rewards(r):
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

running_reward = None
reward_sum = 0
episode_number = 0

rewards_received = []
log_p_actions = []

observation = env.reset()
prev_x = preprocess(observation)
while True:
    if render:
        env.render()

    cur_x = preprocess(observation)
    x = cur_x - prev_x
    prev_x = cur_x

    action, log_p_action = select_action(x)

    log_p_actions.append(log_p_action)

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    rewards_received.append(reward)

    if done:
        episode_number += 1

        discounted_rewards = discount_rewards(rewards_received)

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)


        loss = -(discounted_rewards * log_p_actions).sum()
        loss.backward()

        if episode_number % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

            epoch = episode_number // batch_size
            writer.add_scalar('Running Reward', running_reward, epoch)

        if running_reward is None:
            running_reward = reward_sum
        else:
            running_reward = 0.99 * running_reward + 0.01 * reward_sum

        print(f'episode {episode_number} finished with reward sum = {reward_sum}')

        rewards_received, log_p_actions = [], []
        reward_sum = 0
        observation = env.reset()
        prev_x = preprocess(observation)

