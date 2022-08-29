from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hiddens):
        super(Net, self).__init__()
        self.h = nn.Linear(n_states, n_hiddens)
        self.policy = nn.Sequential(
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_actions),
            nn.Softmax()
        )
        self.value = nn.Sequential(
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, 1)
        )
    
    def forward(self, x):
        x = self.h(x)
        policy = self.policy(x)
        value = self.value(x)
        return value, policy

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hiddens):
        super(PolicyNet, self).__init__()
        self.h = nn.Linear(n_states, n_hiddens)
        self.policy = nn.Sequential(
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.h(x)
        x = self.policy(x)
        return x

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, shared_re):
        super(ValueNet, self).__init__()
        self.h = shared_re
        self.value = nn.Sequential(
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, 1)
        )
    
    def forward(self, x):
        x = self.h(x)
        x = self.value(x)
        return x


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
        reward = reward / np.exp((np.array(range(len(self.buffer)))*np.log(self.config.gamma))) 
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
        self.policy_net = PolicyNet(self.n_states, self.n_actions, config.model.n_hiddens).to(config.device)
        self.value_net = ValueNet(self.config.n_states, config.model.n_hiddens, shared_re=self.policy_net.h).to(config.device)
        self.replay = ReplayBuffer(config)
        self.sample_count = 0
        self.optimizer1 = optim.Adam(self.policy_net.parameters(), lr = self.config.train.lr)
        self.optimizer2 = optim.Adam(self.value_net.parameters(), lr = self.config.train.lr)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr = self.config.train.lr)

    def load_model(self, path):
        states = torch.load(path+self.config.env+".pth") 
        self.policy_net.load_state_dict(states[0])
        self.value_net.load_state_dict(states[1])
        logging.info(f"Loading Model: {path+self.config.env}.pth")

    def save_model(self, path):
        states = [
            self.policy_net.state_dict(),
            self.value_net.state_dict()
        ]
        torch.save(
            obj=states,
            f=path+self.config.env+".pth"
        )

    def update(self):
        device = self.config.device
        state, action, reward = self.replay.sample()
        state, action, reward = state.to(device), action.to(device), reward.to(device)
        value_predict = self.value_net(state)
        target = value_predict - reward
        loss_value = target * value_predict
        decay = torch.tensor(np.exp((np.array(range(len(self.replay)))*np.log(self.config.gamma)))).reshape(target.shape).to(device) 
        #print(decay.shape, target.shape, reward.shape, value_predict.shape)
        #print(decay)
        probs = self.policy_net(state)
        m = Categorical(probs)
        #print(reward.shape, m.log_prob(action.squeeze(1)).shape, reward.shape)
        loss_policy = m.log_prob(action.squeeze(1)).unsqueeze(1) * target * decay
        #print(loss.shape)
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        loss_value.sum().backward(retain_graph=True)
        loss_policy.sum().backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.train.grad_clip)
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.train.grad_clip)
        self.optimizer2.step()
        self.optimizer1.step()


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
