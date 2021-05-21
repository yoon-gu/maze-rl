from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mapf import Mapf, Obstacle, People
from mesa.visualization.modules import ChartModule

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

ws_positions = [(0, 2), (0, 8)]
ws_obstacles = []
for x, y in ws_positions:
    ws_obstacles += [(x+1, y-1), (x+1, y), (x+1, y+1)]

grid = CanvasGrid(agent_portrayal, width, height, 500, 500)

chart = ChartModule([{"Label": "Collision",
                    "Color": "Red"}],
                    data_collector_name='datacollector')

server = ModularServer( Mapf,
                        [grid, chart],
                        "Money Model",
                        {"width":width, "height":height,
                        "start_candidates": [(10, 5)],
                        "goal_candidates": ws_positions,
                        "obstacle_positions": obstacle_positions + ws_obstacles,
                        "num_agents": 5
                        })
server.port = 8522 # The default
server.launch()
