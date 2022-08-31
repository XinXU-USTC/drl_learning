import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from torch.distributions import Categorical

class ReplayBuffer():
    def __init__(self, config):
        self.buffer = []
        self.config = config
    
    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, next_action, done):
        self.buffer.append((state, action, reward, next_state, next_action, done))

    def sample(self):
        batch = self.buffer
        state, action, reward, next_state, next_action, done = zip(*batch)
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(action).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        next_action = torch.tensor(next_action).unsqueeze(1)
        done = torch.tensor(done).unsqueeze(1)
        return state, action, reward, next_state, next_action, done

    def reset(self):
        self.buffer = []

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, n_hiddens):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_hiddens):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class Agent():
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.n_states, self.n_actions = config.n_states, config.n_actions
        self.critic_net = Critic(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.target_net = Critic(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.actor_net = Actor(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.syn_target()
        self.sample_count = 0
        self.memory = ReplayBuffer(config)
        self.optimizer1 = optim.Adam(self.critic_net.parameters(), lr = self.config.train.lr)
        self.optimizer2 = optim.Adam(self.actor_net.parameters(), lr = self.config.train.lr)
        self.loss = nn.MSELoss()

    def load_model(self, path):
        states = torch.load(path+self.config.env+".pth")
        self.actor_net.load_state_dict(states[0])
        self.critic_net.load_state_dict(states[1])
        logging.info(f"Loading Model: {path+self.config.env}.pth")

    def save_model(self, path):
        states = [
            self.actor_net.state_dict(),
            self.critic_net.state_dict()
        ]
        torch.save(
            obj=states,
            f=path+self.config.env+".pth"
        )

    def update(self):
        device = self.config.device
        #state = torch.tensor(np.array(state), dtype=torch.float).reshape(1, -1)
        #action = torch.tensor(action).reshape(1, 1)
        #next_action = torch.tensor(next_action).reshape(1, 1)
        #reward = torch.tensor(reward, dtype=torch.float).reshape(1, 1)
        #next_state = torch.tensor(np.array(next_state), dtype=torch.float).reshape(1, -1)
        #done = torch.tensor(done, dtype=torch.float).reshape(1, 1)
        state, action, reward, next_state, next_action, done = self.memory.sample()
        state, action, reward, next_state, next_action, done = state.to(device), action.to(device), reward.to(device), next_state.to(device), next_action.to(device), done.to(device)
        #print(state.shape, action.shape, reward.shape, next_state.shape, next_action.shape, done.shape)
        q_value = self.critic_net(state).gather(dim=1, index=action)
        next_q_value = self.target_net(next_state).gather(dim=1, index=next_action).detach()
        q_target = reward + self.config.gamma * next_q_value * (~done)
        loss1 = self.loss(q_value, q_target)
        self.optimizer1.zero_grad()
        loss1.sum().backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.config.train.grad_clip)
        self.optimizer1.step()
        probs = self.actor_net(state)
        m = Categorical(probs)
        loss2 = -m.log_prob(action.squeeze(1)).unsqueeze(1)*(q_value.detach())
        #print(loss1.sum()==loss1)
        self.optimizer2.zero_grad()
        loss2.sum().backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.config.train.grad_clip)
        self.optimizer2.step()

    def sample_action(self, state):
        return self.predict_action(state)

    def predict_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, device = self.config.device, dtype=torch.float).unsqueeze(0)
            probs = self.actor_net(state)
            m = Categorical(probs)
            action = m.sample().item()
        return action

    def syn_target(self):
        self.target_net.load_state_dict(self.critic_net.state_dict())