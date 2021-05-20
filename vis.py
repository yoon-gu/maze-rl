from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from maze import Maze, Obstacle, People

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

grid = CanvasGrid(agent_portrayal, 6, 6, 500, 500)

obstacle_positions = [(2, 2), (5, 3), (1, 3), (3,5)]

server = ModularServer( Maze,
                        [grid],
                        "Money Model",
                        {"width":6, "height":6,
                        "start": (0, 0),
                        "obstacle_positions": obstacle_positions
                        })
server.port = 8521 # The default
server.launch()
