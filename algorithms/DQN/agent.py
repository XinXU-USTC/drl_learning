import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

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
        return x

class Agent():
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.n_states, self.n_actions = config.n_states, config.n_actions
        self.poily_net = Net(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.target_net = Net(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.sample_count = 0

    def load_model(self, path):
        self.poily_net = torch.load(path+self.config.env+".pt")
        logging.info(f"Loading Model: {path+self.config.env}.pt")

    def save_model(self, path):
        torch.save(
            obj=self.Q_table,
            f=path+self.config.env+".pkl",
        )

    def update(self, state, action, reward, next_state, done):
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.config.gamma * np.max(self.Q_table[str(next_state)])
        Q_predict = self.Q_table[str(state)][action]
        self.Q_table[str(state)][action] += self.config.train.lr * (Q_target - Q_predict)

    def sample_action(self, state, episode):
        self.sample_count += 1
        epsilon = self.config.greedy.epsilon_end + (self.config.greedy.epsilon_start - self.config.greedy.epsilon_end)\
             * np.exp(-1. * self.sample_count/ self.config.greedy.epsilon_decay)
        #epsilon = 1./(episode + 1)
        if np.random.uniform(0, 1) > epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.n_actions)
        return action

    def predict_action(self, state):
        action = np.argmax(self.Q_table[str(state)])
        return action