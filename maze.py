import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
import gym
from gym import error, spaces, utils
from gym.utils import seeding

ACTIONS = [ (0,  1), 
            (0, -1), 
            ( 1, 0), 
            (-1, 0), 
            ( 0, 0) ]

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.model = Maze(4, 1)
        self.the_agent = self.model.schedule.agents[0]
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_size = 2
        self.observation_space = spaces.Box(
            low=0, high=max(self.model.width, self.model.height), shape=(self.observation_size,))
        self.counter = 0

    def step(self, action):
        self.counter += 1
        info = {}
        next_pos = np.array(self.state) + np.array(ACTIONS[action])
        self.the_agent = self.model.schedule.agents[0]
        self.the_agent.next_pos = tuple(next_pos)
        self.model.step()
        reward = self.model.get_reward()
        done = not self.model.running
        if self.counter > 100:
            done = True
        state = np.array(self.the_agent.pos)
        self.state = state
        return self.state, reward, done, info

    def reset(self):
        self.model = Maze(4, 1)
        state = np.array(self.the_agent.pos)
        self.state = state
        return self.state

class People(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.done = False
        self.reward = None
        self.reward_sum = 0
        self.next_pos = None

    def step(self):
        self.move()

    def move(self):
        self.reward = 0.0
        if not self.done:
            possible_steps = self.model.grid.get_neighborhood(
                    self.pos,
                    moore=False,
                    include_center=False
                )
            if self.next_pos is None:
                new_position = self.random.choice(possible_steps)
                self.model.grid.move_agent(self, new_position)
                self.reward = -1
            else:
                if self.next_pos in possible_steps:
                    self.model.grid.move_agent(self, self.next_pos)
                    self.reward = -1
                else:
                    self.reward = -3

            if (self.pos[0] == self.model.goal[0]) and (self.pos[1] == self.model.goal[1]):
                self.done = True
                self.reward = 20
        self.reward_sum += self.reward

class Maze(Model):
    def __init__(self, width=4, height=1, start=(0,0)):
        super().__init__()
        self.width = width
        self.height = height
        self.goal = (width - 1, height - 1)
        # Create agents
        people = People(self.next_id(), self)
        self.schedule = RandomActivation(self)
        self.schedule.add(people)
        self.grid = MultiGrid(width, height, False)
        self.grid.place_agent(people, start)
        self.running = True
        
    def step(self):
        self.schedule.step()
        if all([a.done for a in self.schedule.agents]):
            self.running = False
        self.reward = 0.0
        for agent in self.schedule.agents:
            self.reward += agent.reward / len(self.schedule.agents)
    
    def get_reward(self):
        return self.reward