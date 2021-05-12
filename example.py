from maze import MazeEnv, ACTIONS
import numpy as np

env = MazeEnv()
state = env.reset()

num_episode = 20
done = False
print(0, state, 0.0)
for e in range(1, 1 + num_episode):
    if not done:
        action = np.random.choice(5)
        new_state, reward, done, _ = env.step(action)
        print(e, new_state, reward, ACTIONS[action])