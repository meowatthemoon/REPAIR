from Breakout_environment import Game
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


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


def inc_feedback_prob(ball_y):
    return max(0, -1 / 8 * ball_y + 1)


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


def state_action_feedback_to_coord(state, action, feedback):
    index = 0
    ball_x = state[0]
    ball_y = state[1]
    ball_speed_x = state[2]
    ball_speed_y = state[3]
    paddle_x = state[4]

    index += 12000 * (int(feedback == 1))
    index += 4000 * action
    index += 2000 * ball_speed_x
    index += 1000 * ball_speed_y
    index += 100 * ball_x
    index += 10 * ball_y
    index += 1 * paddle_x
    return index


def trust(feedback, triplet_coord):
    global R_max

    min = np.min(R_max)
    max = np.max(R_max)
    value = R_max[triplet_coord]

    # print(f"Value = {value}, Min = {min}, Max = {max}")

    if feedback > 0:
        trust = (value - min) / (max - min)
    else:
        trust = 1 - (value - min) / (max - min)
    return trust


# Init environment
env = Game()

# Create Q table
action_space_size = env.n_actions  # 3
state_space_size = env.n_states_disc  # 4000
q_table = np.zeros((state_space_size, action_space_size))  # n_states x n_actions

# Q-Learning Hyper parameters
num_episodes = 100000
max_steps_per_episode = 1000000

learning_rate = 0.1  # alpha
discount_rate = 0.99  # gamma

exploration_rate = 1  # epsilon
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# REPAIR parameters
R_max = np.array([-10000 for x in range(action_space_size * state_space_size * 2)])  # feedback is either 1 or -1
t_min, t_max = 0.4, 0.6

# Algorithm
rewards_all_episodes = []

# Stats
deleted_all_episodes = []
corrected_all_episodes = []
incorrect_all_episodes = []
for episode in range(num_episodes):
    # Repair episode initialization
    R_trajectory = 0
    trajectory = []

    # Q Learning episode initialization
    state_v = env.reset()
    state_c = state_to_coord(state_v)

    done = False

    print(f"{episode + 1} / {num_episodes}")

    # Play episode
    for step in range(max_steps_per_episode):
        # Exploration vs Exploitation
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state_c, :])
        else:
            action = np.random.choice([x for x in range(action_space_size)])

        # Step
        new_state_v, reward, done, feedback = env.step(action)

        # Repair episode updates
        R_trajectory += reward

        # Potentially make feedback incorrect
        ball_y = new_state_v[1]
        prob_inc = inc_feedback_prob(ball_y)
        if random.uniform(0, 1) < prob_inc:
            feedback = feedback * -1
        trajectory.append((state_v, action, feedback, reward, new_state_v))

        # Update state
        state_v = new_state_v
        state_c = state_to_coord(new_state_v)

        if done == True:
            break
    num_transitions = 0
    num_corrected = 0
    num_incorrected = 0
    num_deleted = 0
    # End of episode
    for (state_v, action, feedback, reward, new_state_v) in trajectory:
        triplet_coord = state_action_feedback_to_coord(state_v, action, feedback)

        # Update R_max list
        R_max[triplet_coord] = max(R_max[triplet_coord], R_trajectory)

        # Obtain trust value
        trust_ = trust(feedback=feedback, triplet_coord=triplet_coord)

        num_transitions += 1
        # Filter the feedback
        if trust_ <= t_min:
            feedback = -feedback
            num_corrected += 1
        elif trust_ >= t_max:
            feedback = feedback
            num_incorrected += 1
        else:
            feedback = 0
            num_deleted += 1

        # Update Q table
        state_c = state_to_coord(state_v)
        new_state_c = state_to_coord(new_state_v)
        q_table[state_c, action] = q_table[state_c, action] * (1 - learning_rate) + \
                                   learning_rate * (reward + feedback + discount_rate * np.max(q_table[new_state_c, :]))

    # Update exploration rate
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(R_trajectory)

    deleted_all_episodes.append(num_deleted / num_transitions)
    corrected_all_episodes.append(num_corrected / num_transitions)
    incorrect_all_episodes.append(num_incorrected / num_transitions)

# Stats
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
mean_scores = []
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000
    mean_scores.append(sum(r / 1000))

with open("repair_table.data", 'wb') as file:
    pickle.dump(q_table, file)

plotLearning([x for x in range(0, num_episodes, 1000)], mean_scores, "repair_2.jpg")

# Stats corrected
corrected_per_thousand_episodes = np.split(np.array(corrected_all_episodes), num_episodes / 1000)
count = 1000
mean_scores = []
for r in corrected_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000
    mean_scores.append(sum(r / 1000))

plotLearning([x for x in range(0, num_episodes, 1000)], mean_scores, "repair_correct.jpg")

# Stats incorrect
incorrect_per_thousand_episodes = np.split(np.array(incorrect_all_episodes), num_episodes / 1000)
count = 1000
mean_scores = []
for r in incorrect_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000
    mean_scores.append(sum(r / 1000))

plotLearning([x for x in range(0, num_episodes, 1000)], mean_scores, "repair_incorrect.jpg")

# Stats deleted
deleted_per_thousand_episodes = np.split(np.array(deleted_all_episodes), num_episodes / 1000)
count = 1000
mean_scores = []
for r in deleted_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000
    mean_scores.append(sum(r / 1000))

plotLearning([x for x in range(0, num_episodes, 1000)], mean_scores, "repair_deleted.jpg")
