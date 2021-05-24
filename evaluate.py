from mapf import MultipleAgentsMaze
from tqdm import tqdm
import pickle
import numpy as np

num_agents = 10
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

with open('benchmark-{}x{}.pkl'.format(width,height), 'rb') as f:
    Q = pickle.load(f)

ACTIONS = [ (0,  1),  # RIGHT
            (0, -1),  # LEFT
            ( 1, 0),  # DOWN
            (-1, 0)]  # UP

def policy(pos):

    try:
        action = np.argmax(Q[pos])
    except:
        print(pos)
    next_pos = tuple(np.array(pos) + np.array(ACTIONS[action]))
    return next_pos

def random_policy(pos):

    try:
        action = np.random.choice(len(ACTIONS))
    except:
        print(pos)
    next_pos = tuple(np.array(pos) + np.array(ACTIONS[action]))
    return next_pos

num_episodes = 1000
avg_collision = 0.0
avg_reward_sum = 0.0
np.random.seed(1234)
for _ in tqdm(range(num_episodes)):
    model = MultipleAgentsMaze( width, height, 
                            start_positions, ws_positions[1:], 
                            obstacle_positions + ws_obstacles, 
                            num_agents, policy)
    reward_sum = 0.0
    while model.running:
        model.step()
        reward_sum += model.reward
    avg_reward_sum += reward_sum / num_episodes
    avg_collision += model.collision_count / num_episodes

print("Monte-Carlo Control")
print(avg_reward_sum, avg_collision)

avg_collision = 0.0
avg_reward_sum = 0.0
np.random.seed(1234)
for _ in tqdm(range(num_episodes)):
    model = MultipleAgentsMaze( width, height, 
                            start_positions, ws_positions[1:], 
                            obstacle_positions + ws_obstacles, 
                            num_agents, random_policy)
    reward_sum = 0.0
    while model.running:
        model.step()
        reward_sum += model.reward
    avg_reward_sum += reward_sum / num_episodes
    avg_collision += model.collision_count / num_episodes

print("Random Walk")
print(avg_reward_sum, avg_collision)
