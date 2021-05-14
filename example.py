from maze import MazeEnv, ACTIONS

obstacle_positions = [(2, 2), (5, 3), (1, 3)]

env = MazeEnv(6, 6, (0, 0), obstacle_positions)
state = env.reset()

num_episode = 20
done = False
print(0, state, 0.0)
e = 1
while not done:
    action = env.action_space.sample()
    new_state, reward, done, _ = env.step(action)
    print(e, new_state, reward, ACTIONS[action])
    e += 1