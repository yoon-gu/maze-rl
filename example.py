from maze import MazeEnv

env = MazeEnv()
state = env.reset()

num_episode = 20
done = False
for e in range(num_episode):
    if not done:
        new_state, reward, done, _ = env.step(0)
        print(e, new_state, reward)