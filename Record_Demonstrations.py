import pickle
from Breakout_environment import Game
import numpy as np
import time


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


def read_table(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


# Demonstration Data
demonstrations = []
min_transitions = 50000
# Init environment
env = Game()

# Load Q table
q_table = read_table("repair_table.data")

games = 0
while len(demonstrations) < min_transitions:
    done = False
    accumulate_rewards = 0
    state = env.reset()
    state = state_to_coord(state)
    while not done:
        action = np.argmax(q_table[state, :])

        # Step
        new_state, reward, done, feedback = env.step(action)
        new_state = state_to_coord(new_state)

        demonstrations.append([state, action, reward + feedback, new_state, done])

        # state = new state and accumulate rewards
        state = new_state
        accumulate_rewards += reward

        env.draw()
        #time.sleep(0.01)

        #print(feedback)

        if done == True:
            break
    games +=1
    print(f"{games} : {accumulate_rewards}")


with open("DQfD/demonstration_19_games_50000transitions.data", 'wb') as file:
    pickle.dump(demonstrations, file)
