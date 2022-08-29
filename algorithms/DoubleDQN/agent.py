import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import random

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

class ReplayBuffer():
    def __init__(self, config):
        self.size = config.train.buffer_size
        self.position = 0
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(action).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(1)
        return state, action, reward, next_state, done

class Agent():
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.n_states, self.n_actions = config.n_states, config.n_actions
        self.policy_net = Net(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.target_net = Net(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.syn_target()
        self.replay = ReplayBuffer(config)
        self.sample_count = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.config.train.lr)
        self.loss = nn.MSELoss()

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path+self.config.env+".pth"))
        #for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
        #     param.data.copy_(target_param.data)
        logging.info(f"Loading Model: {path+self.config.env}.pth")

    def save_model(self, path):
        torch.save(
            obj=self.policy_net.state_dict(),
            f=path+self.config.env+".pth"
        )

    def update(self):
        if len(self.replay) < self.config.train.batch_size:
            return
        device = self.config.device
        state, action, reward, next_state, done = self.replay.sample(self.config.train.batch_size)
        state, action, reward, next_state, done = state.to(device), action.to(device), reward.to(device), next_state.to(device), done.to(device)
        q_value = self.policy_net(state).gather(dim=1, index=action)
        next_q_value = self.policy_net(next_state).detach()
        next_max_q_value = self.target_net(next_state).gather(dim=1, index=next_q_value.max(1)[1].unsqueeze(1))
        q_target = reward + self.config.gamma * next_max_q_value * (1 - done)
        #print(q_value.shape, next_max_q_value.shape)
        loss = self.loss(q_value, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.train.grad_clip)
        self.optimizer.step()
        #for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
        #    target_param = target_param * (1 - self.config.train.tau) + self.config.train.tau * param


    def sample_action(self, state, episode):
        self.sample_count += 1
        epsilon = self.config.greedy.epsilon_end + (self.config.greedy.epsilon_start - self.config.greedy.epsilon_end)\
             * np.exp(-1. * self.sample_count/ self.config.greedy.epsilon_decay)
        #epsilon = 1./(episode + 1)
        if np.random.uniform(0, 1) > epsilon:
            with torch.no_grad():
                action = self.predict_action(state)
        else:
            action = np.random.choice(self.n_actions)
        return action

    def predict_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, device = self.config.device, dtype=torch.float).unsqueeze(0)
            action = self.policy_net(state).max(1)[1].item()
        return action

    def syn_target(self):
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)