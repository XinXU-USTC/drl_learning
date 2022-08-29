import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import random
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hiddens):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class ReplayBuffer():
    def __init__(self, config):
        self.buffer = []
        self.config = config
    
    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self):
        batch = self.buffer
        state, action, reward = zip(*batch)
        #print(reward)
        reward = np.array(reward) * np.exp((np.array(range(len(self.buffer)))*np.log(self.config.gamma)))
        reward = np.flip(reward)
        reward = np.cumsum(reward)
        reward = np.flip(reward)
        reward = np.ascontiguousarray(reward)
        #print(len(self.buffer))
        #print(reward.shape)
        #reward = reward / np.exp((np.array(range(len(self.buffer)))*np.log(self.config.gamma))) 
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(action).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1)
        #print(reward)
        return state, action, reward

    def reset(self):
        self.buffer = []

class Agent():
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.n_states, self.n_actions = config.n_states, config.n_actions
        self.policy_net = Net(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.replay = ReplayBuffer(config)
        self.sample_count = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.config.train.lr)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr = self.config.train.lr)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path+self.config.env+".pth"))
        logging.info(f"Loading Model: {path+self.config.env}.pth")

    def save_model(self, path):
        torch.save(
            obj=self.policy_net.state_dict(),
            f=path+self.config.env+".pth"
        )

    def update(self):
        device = self.config.device
        state, action, reward = self.replay.sample()
        state, action, reward = state.to(device), action.to(device), reward.to(device)
        probs = self.policy_net(state)
        m = Categorical(probs)
        #print(reward.shape, m.log_prob(action.squeeze(1)).shape, reward.shape)
        loss = -m.log_prob(action.squeeze(1)).unsqueeze(1) * reward
        #print(loss.shape)
        self.optimizer.zero_grad()
        loss.sum().backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.train.grad_clip)
        self.optimizer.step()


    def sample_action(self, state):
        return self.predict_action(state)

    def predict_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, device = self.config.device, dtype=torch.float).unsqueeze(0)
            probs = self.policy_net(state)
            m = Categorical(probs)
            #action = m.sample().to('cpu').numpy().astype(int)[0]
            action = m.sample().item()
        return action
