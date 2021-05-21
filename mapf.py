import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from mesa.datacollection import DataCollector

ACTIONS = [ (0,  1),  # RIGHT
            (0, -1),  # LEFT
            ( 1, 0),  # DOWN
            (-1, 0)]  # UP

class Mapf(Model):
    def __init__(self, width, height,
                start_candidates, goal_candidates,
                obstacle_positions,
                num_agents):
        super().__init__()
        self.width = width
        self.height = height
        self.start_candidates = start_candidates
        self.goal_candidates = goal_candidates
        self.obstacle_positions = obstacle_positions
        self.collision_count = 0

        assert len(set(start_candidates).intersection(set(obstacle_positions))) == 0
        assert len(set(goal_candidates).intersection(set(obstacle_positions))) == 0

        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, False)

        self.datacollector = DataCollector(
            model_reporters={"Collision": lambda model: model.collision_count},
            agent_reporters={})

        # Create agents
        self.agents = []
        for _ in range(num_agents):
            start = self.start_candidates[np.random.choice(len(self.start_candidates))]
            goal = self.goal_candidates[np.random.choice(len(self.goal_candidates))]
            people = People(self.next_id(), self, start, goal)

            self.schedule.add(people)
            self.grid.place_agent(people, start)
            self.agents.append(people)
        self.running = True

        # Create obstacles
        for x, y in self.obstacle_positions:
            o = Obstacle(self.next_id(), self)
            self.schedule.add(o)
            self.grid.place_agent(o, (x, y))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

        agent_positions = set([a.pos for a in self.agents])
        for pos in agent_positions:
            agents_at_the_cell = self.grid.get_cell_list_contents(pos)
            if len(agents_at_the_cell) > 1:
                self.collision_count += 1

        if all([a.done for a in self.schedule.agents if type(a) is People]):
            self.running = False
        self.reward = 0.0
        for agent in self.schedule.agents:
            if type(agent) is People:
                self.reward += agent.reward

    def get_reward(self):
        return self.reward


class People(Agent):
    def __init__(self, unique_id, model, start, goal):
        super().__init__(unique_id, model)
        self.done = False
        self.start = start
        self.goal = goal
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

            if (self.pos[0] == self.goal[0]) and (self.pos[1] == self.goal[1]):
                self.done = True
                self.reward = 20
        self.reward_sum += self.reward

class Obstacle(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)