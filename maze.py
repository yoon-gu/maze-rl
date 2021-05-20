import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
import gym
from gym import error, spaces, utils
from gym.utils import seeding

ACTIONS = [ (0,  1),  # RIGHT
            (0, -1),  # LEFT
            ( 1, 0),  # DOWN
            (-1, 0)]  # UP

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, start=(0,0), goal=None, obstacle_positions=[]):
        self.width = width
        self.height = height
        self.start = start
        self.obstacle_positions = obstacle_positions
        if goal is None:
            self.goal = (width - 1, height - 1)
        else:
            self.goal = goal
        self.model = Maze(width, height, start, goal, obstacle_positions)
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

        state = self.the_agent.pos
        self.state = state
        return self.state, reward, done, info

    def reset(self):
        self.model = Maze(self.width, self.height, self.start, self.goal, self.obstacle_positions)
        self.the_agent = self.model.schedule.agents[0]
        state = self.the_agent.pos
        self.state = state
        return self.state
    
    def render(self):
        world = 255 * np.ones((self.width, self.height, 3), dtype=np.uint8)
        start_x, start_y = self.model.start
        goal_x, goal_y = self.model.goal
        world[start_x, start_y, :] = [255, 128, 0]
        world[goal_x, goal_y, :] = [0, 0, 255]
        for agent in self.model.schedule.agents:
            x, y = agent.pos
            if type(agent) is People:
                world[x, y, :] = [128, 128, 128]
            elif type(agent) is Obstacle:
                world[x, y, :] = [0, 0, 0]

        return world

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

            possible_steps = list(set(possible_steps).difference(set(self.model.obstacle_positions)))
            if self.next_pos is None:
                new_position = self.random.choice(possible_steps)
                self.model.grid.move_agent(self, new_position)
                self.reward = -1
            else:
                if self.next_pos in possible_steps:
                    self.model.grid.move_agent(self, self.next_pos)
                    self.reward = -1
                else:
                    self.reward = -5

            if (self.pos[0] == self.model.goal[0]) and (self.pos[1] == self.model.goal[1]):
                self.done = True
                self.reward = 20
        self.reward_sum += self.reward

class Obstacle(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Maze(Model):
    def __init__(self, width, height, start=(0, 0), goal=None, obstacle_positions=[]):
        super().__init__()
        self.width = width
        self.height = height
        self.start = start
        if goal is None:
            self.goal = (width - 1, height - 1)
        else:
            self.goal = goal
        self.obstacle_positions = obstacle_positions
        # Create agents
        people = People(self.next_id(), self)
        self.schedule = RandomActivation(self)
        self.schedule.add(people)
        self.grid = MultiGrid(width, height, False)
        self.grid.place_agent(people, start)
        self.running = True
        
        # Create obstacles
        for x, y in self.obstacle_positions:
            o = Obstacle(self.next_id(), self)
            self.schedule.add(o)
            self.grid.place_agent(o, (x, y))

    def step(self):
        self.schedule.step()
        if all([a.done for a in self.schedule.agents if type(a) is People]):
            self.running = False
        self.reward = 0.0
        for agent in self.schedule.agents:
            if type(agent) is People:
                self.reward += agent.reward
    
    def get_reward(self):
        return self.reward
