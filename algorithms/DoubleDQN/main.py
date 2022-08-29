import gym
import numpy as np
import logging
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import random
from algorithms.DoubleDQN.agent import Agent
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 4, 5, 6, 7, 8"


class Runner():
    def __init__(self, args, config):
        self.env = self.set_env(config.env)
        self.env.seed(args.seed)
        self.args = args
        self.config = config
        self.n_states = self.env.observation_space.shape[0]
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

        agent.policy_net = nn.DataParallel(agent.policy_net, device_ids=range(config.train.device_ids))
        agent.target_net = nn.DataParallel(agent.target_net, device_ids=range(config.train.device_ids))
        
        for episode in tqdm(range(config.train.n_episode), desc = 'Episodes'):
            ep_reward, ep_step = 0, 0
            state = env.reset()
            while True:
                action = agent.sample_action(state, episode)
                next_state, reward, done, _ = env.step(action)
                agent.replay.push(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                ep_reward += reward
                ep_step += 1
                #logging.info(
                #    f"episode: {episode}, step: {ep_step}, reward: {reward}, accum_reward: {ep_reward}"
                #)
                if done:
                    break
            if (episode + 1) % self.config.train.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
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
        agent.policy_net = nn.DataParallel(agent.policy_net, device_ids=range(config.train.device_ids))
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


