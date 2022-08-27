import turtle
import gym
import numpy as np
import logging
from tqdm import tqdm
import os

from algorithms.sarsa.agent import Agent


class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.t = None
        self.unit = 50
        self.max_x = 12
        self.max_y = 4

    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            for i in range(1, self.max_x - 1):
                self.draw_box(i, 0, 'black')
            self.draw_box(self.max_x - 1, 0, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)

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
        if env_name == 'CliffWalking-v0':
            env = CliffWalkingWapper(env)
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
                next_action = agent.sample_action(state, episode)
                agent.update(state, action, reward, next_state, next_action, done)
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
        #state = env.reset()
        agent.load_model(args.log_path+"/")
        for episode in tqdm(range(config.test.n_episode), desc = 'Episodes'):
        #for episode in range(config.test.n_episode):
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
                #if ep_step == 10:
                #    break
            tb_logger.add_scalar("reward", ep_reward, global_step=episode)
            logging.info(
                f"episode: {episode}, steps: {ep_step}, reward: {ep_reward}"
            ) 
        logging.info(f"End Training! {config.test.n_episode} Episodes!") 


