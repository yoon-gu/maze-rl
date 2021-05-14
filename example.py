from maze import MazeEnv, ACTIONS
import numpy as np

obstacle_positions = [(2, 2), (5, 3), (1, 3)]

env = MazeEnv(6, 6, (0, 0), obstacle_positions)
state = env.reset()

num_episode = 20
done = False
print(0, state, 0.0)
for e in range(1, 1 + num_episode):
    if not done:
        action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action)
        print(e, new_state, reward, ACTIONS[action])