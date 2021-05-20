# maze-rl
```python
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
ACTION_LABELS = ['→', '←', '↓', '↑']
width, height = 8, 8


# %%
obstacle_positions = [(2, 2), (5, 3), (1, 3), (3,5)]


# %%
from maze import MazeEnv, ACTIONS
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from IPython import display
import numpy as np
from tqdm import tqdm

filename = 'imagedraw.gif'
images = []
env = MazeEnv(width, height, (0, 0), obstacle_positions)
state = env.reset()
done = False
reward_sum = 0.0

img = env.render()
im = Image.fromarray(img)
im = im.resize((400, 400), resample=0)
images.append(im)

while not done:
    action = np.random.choice(len(ACTIONS))
    state, reward, done, _ = env.step(action)
    # print(state, reward, done)
    reward_sum += reward
    img = env.render()
    im = Image.fromarray(img)
    im = im.resize((400, 400), resample=0)
    images.append(im)
    
images[0].save(filename,
               save_all=True, append_images=images[1:],
               optimize=False, duration=40, loop=0)


# %%
display.Image(filename)


# %%
from collections import defaultdict

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

def generate_episode_from_Q(env, Q, epsilon):
    nA = env.action_space.n
    episode = []
    state = env.reset()
    done = False
    while not done:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA))                                     if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode

def update_Q(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]] 
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q


# %%
env = MazeEnv(width, height, (0, 0), obstacle_positions)
num_episodes = 50000
alpha = 0.05

gamma= 0.9
eps_start=1.0
eps_decay=.99999
eps_min=0.01

# Main loop
nA = env.action_space.n

# Initialization
Q = defaultdict(lambda: np.zeros(nA))
epsilon = eps_start
# loop over episodes
for i_episode in tqdm(range(num_episodes)):
    epsilon = max(epsilon * eps_decay, eps_min)
    episode = generate_episode_from_Q(env, Q, epsilon)
    Q = update_Q(env, episode, Q, alpha, gamma)
policy = dict((k, np.argmax(v)) for k, v in Q.items())


# %%
import pandas as pd
pd.DataFrame(Q).T.sort_index()


# %%
env = MazeEnv(width, height, (0, 0), obstacle_positions)
episode = generate_episode_from_Q(env, Q, 0.01)
print(episode)


# %%
env = MazeEnv(width, height, (0, 0), obstacle_positions)
episode = generate_episode_from_Q(env, Q, 0.00)
plt.imshow(env.render())
for state, action, reward in episode:
    x, y = state
    plt.text(y, x, str(ACTION_LABELS[action]),
    horizontalalignment='center',
    verticalalignment='center')
plt.show()


# %%
env = MazeEnv(width, height, (0, 0), obstacle_positions)
env.reset()
plt.imshow(env.render())
for x in range(width):
    for y in range(height):
        action = policy.get((x, y), None)
        if action is not None:
            plt.text(y, x, str(ACTION_LABELS[action]),
                horizontalalignment='center',
                verticalalignment='center')
plt.show()


# %%
env = MazeEnv(width, height, (0, 3), obstacle_positions)
episode = generate_episode_from_Q(env, Q, 0.00)
plt.imshow(env.render())
for state, action, reward in episode:
    x, y = state
    plt.text(y, x, str(ACTION_LABELS[action]),
    horizontalalignment='center',
    verticalalignment='center')
plt.show()


# %%
print(episode)


# %%
env = MazeEnv(width, height, (5, 2), obstacle_positions)
episode = generate_episode_from_Q(env, Q, 0.00)
print(episode, sum([r for _, _, r in episode]))


# %%




```
