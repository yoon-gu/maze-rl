from utils import approximate_q_function, get_episode, eval_policy, mc_policy
from maze import MazeEnv
from pprint import pprint

width, height = 6, 6
obstacle_positions = [(2, 2), (5, 3), (1, 3)]
env = MazeEnv(width, height, (0, 0), obstacle_positions)
Q, N = approximate_q_function(env, 5000)

episode = get_episode(env, Q)
pprint(episode)

avg, _ = eval_policy(env, lambda s: mc_policy(env, s, Q, eps=1.0), 100)
avg, _ = eval_policy(env, lambda s: mc_policy(env, s, Q, eps=0.01), 100)
avg, _ = eval_policy(env, lambda s: mc_policy(env, s, Q, eps=0.00), 100)