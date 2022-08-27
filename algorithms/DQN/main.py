import gym
import numpy as np
import logging
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from algorithms.QLearning.agent import Agent

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
        pass

class Runner():
    def __init__(self, args, config):
        self.env = self.set_env(config.env)
        self.args = args
        self.config = config
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        config.n_actions = self.n_actions
        config.n_states = self.n_states
        self.agent = Agent(args, config)

    def set_env(self, env_name):
        env = gym.make(env_name)
        return env

    def train(self):
        args, config = self.args, self.config
        agent, env = self.agent, self.env
        tb_logger = config.tb_logger
        logging.info(f"Start Training! Env: {config.env}, Algorithm: {args.alg}")
        for episode in tqdm(range(config.train.n_episode), desc = 'Episodes'):
            ep_reward, ep_step = 0, 0
            state = env.reset()
            while True:
                action = agent.sample_action(state, episode)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                ep_step += 1
                #logging.info(
                #    f"episode: {episode}, step: {ep_step}, reward: {reward}, accum_reward: {ep_reward}"
                #)
                if done:
                    break
            tb_logger.add_scalar("reward", ep_reward, global_step=episode)
            logging.info(
                f"episode: {episode}, steps: {ep_step}, reward: {ep_reward}"
            )
        logging.info(f"End Training! {config.train.n_episode} Episodes!")
        agent.save_model(args.log_path+"/") 

    def test(self):
        args, config = self.args, self.config
        agent, env = self.agent, self.env
        tb_logger = config.tb_logger 
        logging.info(f"Start Testing! Env: {config.env}, Algorithm: {args.alg}")
        agent.load_model(args.log_path+"/")
        for episode in tqdm(range(config.test.n_episode), desc = 'Episodes'):
            ep_reward, ep_step = 0, 0
            state = env.reset()
            while True:
                action = agent.predict_action(state)
                state, reward, done, _ = env.step(action)
                ep_reward += reward
                ep_step += 1
                #logging.info(
                #    f"episode: {episode}, step: {ep_step}, reward: {reward}, accum_reward: {ep_reward}"
                #)
                if done:
                    break
            tb_logger.add_scalar("reward", ep_reward, global_step=episode)
            logging.info(
                f"episode: {episode}, steps: {ep_step}, reward: {ep_reward}"
            ) 
        logging.info(f"End Training! {config.test.n_episode} Episodes!") 


