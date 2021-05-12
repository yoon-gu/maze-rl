from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from maze import Maze

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                "Color": "red",
                "Filled": "true",
                "Layer": 0,
                "Label": str(agent.reward_sum),
                "r": 0.5}
    return portrayal

grid = CanvasGrid(agent_portrayal, 4, 4, 500, 500)

server = ModularServer( Maze,
                        [grid],
                        "Money Model",
                        {"width":4, "height":1})
server.port = 8521 # The default
server.launch()