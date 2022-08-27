import tqdm
import torch
import dill
from collections import defaultdict
import numpy as np
import logging

class Agent():
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.n_states, self.n_actions = config.n_states, config.n_actions
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.sample_count = 0

    def load_model(self, path):
        self.Q_table = torch.load(path+self.config.env+".pkl", pickle_module=dill)
        logging.info(f"Loading Model: {path+self.config.env}.pkl")

    def save_model(self, path):
        torch.save(
            obj=self.Q_table,
            f=path+self.config.env+".pkl",
            pickle_module=dill
        )

    def update(self, state, action, reward, next_state, next_action, done):
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.config.gamma * self.Q_table[str(next_state)][next_action]
        Q_predict = self.Q_table[str(state)][action]
        self.Q_table[str(state)][action] += self.config.train.lr * (Q_target - Q_predict)
        #print(state)

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