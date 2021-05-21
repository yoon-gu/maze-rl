from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mapf import MultipleAgentsMaze, Obstacle, People
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
import pickle
import numpy as np

with open('benchmark-11x11.pkl', 'rb') as f:
    Q = pickle.load(f)

def policy(pos):
    ACTIONS = [ (0,  1),  # RIGHT
            (0, -1),  # LEFT
            ( 1, 0),  # DOWN
            (-1, 0)]  # UP
    try:
        action = np.argmax(Q[pos])
    except:
        print(pos)
    next_pos = tuple(np.array(pos) + np.array(ACTIONS[action]))
    return next_pos

def agent_portrayal(agent):
    if type(agent) is People:
        portrayal = {"Shape": "circle",
                    "Color": "red",
                    "Filled": "true",
                    "Layer": 1,
                    "Label": str(agent.reward_sum),
                    "r": 0.5}
    elif type(agent) is Obstacle:
        portrayal = {"Shape": "rect",
                    "Color": "black",
                    "Filled": "true",
                    "Layer": 0,
                    "w": 0.9,
                    "h": 0.9}
    return portrayal

width, height = 11, 11
obstacle_positions = []
for x in range(4, 10):
    if x % 3 == 0:
        continue
    for y in range(2, 10, 2):
        obstacle_positions.append((x, y))

ws_positions = [(0, 2), (0, 8)] # Goal (0, 2) is not trained
ws_obstacles = []
for x, y in ws_positions:
    ws_obstacles += [(x+1, y-1), (x+1, y), (x+1, y+1)]

all_positions = [(x, y) for x in range(width) for y in range(height)]
start_positions = list(set(all_positions).difference(set(obstacle_positions + ws_obstacles + ws_positions)))
grid = CanvasGrid(agent_portrayal, width, height, 500, 500)

chart = ChartModule([{"Label": "Collision",
                    "Color": "Red"}],
                    data_collector_name='datacollector')

server = ModularServer( MultipleAgentsMaze,
                        [grid, chart],
                        "Maze Model",
                        {"width":width, "height":height,
                        "start_candidates": start_positions,
                        "goal_candidates": [(0, 8)],
                        "obstacle_positions": obstacle_positions + ws_obstacles,
                        "num_agents": UserSettableParameter('slider', 'Agents', 5, 1, 15, 1),
                        "policy": policy
                        })
server.port = 8522 # The default
server.launch()
