from Breakout_environment import Game
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt


def plotLearning(x, scores, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    ax.plot(x, scores, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.set_ylabel("Score", color="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def state_to_coord(state):
    index = 0
    ball_x = state[0]
    ball_y = state[1]
    ball_speed_x = state[2]
    ball_speed_y = state[3]
    paddle_x = state[4]

    index += 2000 * ball_speed_x
    index += 1000 * ball_speed_y
    index += 100 * ball_x
    index += 10 * ball_y

    index += 1 * paddle_x
    return index


# Init environment
env = Game()

# Create Q table
action_space_size = env.n_actions
state_space_size = env.n_states_disc
q_table = np.zeros((state_space_size, action_space_size))  # n_states x n_actions

# Hyper parameters
num_episodes = 100000
max_steps_per_episode = 1000000

learning_rate = 0.1  # alpha
discount_rate = 0.99  # gamma

exploration_rate = 1  # epsilon
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Algorithm
rewards_all_episodes = []

# Q learning
for episode in range(num_episodes):
    state = env.reset()
    state = state_to_coord(state)

    done = False
    rewards_current_episode = 0

    print(f"{episode + 1} / {num_episodes}")

    # Play episode
    for step in range(max_steps_per_episode):
        # Exploration vs Exploitation
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = np.random.choice([x for x in range(action_space_size)])

        # Step
        new_state, reward, done, feedback = env.step(action)
        new_state = state_to_coord(new_state)

        # Update Q table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # state = new state and accumulate rewards
        state = new_state
        rewards_current_episode += reward

        if done == True:
            break
    # End of episode
    # Update exploration rate
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)

# Stats
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
mean_scores = []
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000
    mean_scores.append(sum(r / 1000))

plotLearning([x for x in range(0, num_episodes, 1000)], mean_scores, "q_learning.jpg")

