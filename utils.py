from maze import MazeEnv, ACTIONS
import numpy as np
from tqdm import tqdm

def mc_policy(env, state, Q, eps=0.01):
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    action = policy[state] if np.random.random() > eps else env.action_space.sample()
    return action

def eval_policy(env, policy, num_episodes=100):
    reward_list = []
    for i in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        reward_sum = 0.0
        while not done:
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = new_state

        reward_list.append(reward_sum)

    avg = np.mean(reward_list)
    print("The average cumulative reward {} while {} episodes".format(avg, num_episodes))
    return avg, reward_list

from collections import defaultdict
from pprint import pprint

def approximate_q_function(env, num_episodes = 10000):
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for e_id in tqdm(range(num_episodes)):
        episode = get_episode(env)
        states, actions, rewards = zip(*episode)

        for i, state in enumerate(states):
            returns_sum[state][actions[i]] += sum(rewards[i:])
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]

    return Q, N

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

def get_episode(env, Q=None):
    episode = []
    state = env.reset()
    done = False
    while not done:
        if Q is None:
            action = np.random.choice(len(ACTIONS))
        else:
            action = np.random.choice(len(ACTIONS), p=get_probs(Q[state], 0.01, len(ACTIONS)))
        new_state, reward, done, _ = env.step(action)
        episode.append((tuple(state), action, reward))
        state = new_state
    return episode