import random

import gym
import numpy as np

env = gym.make('FrozenLake8x8-v1', render_mode='human', is_slippery=False)
epoch = 3
state, _ = env.reset()

Q = np.ones([env.observation_space.n, env.action_space.n])
goal_pos = 7, 7

for i in range(epoch):
    done = False
    while not done:
        maksimum = max(Q[state, :])

        max_index = [i for i, x in enumerate(Q[state, :]) if x == maksimum]

        action = random.choice(max_index)

        next_state, reward, terminated, _, info = env.step(action)

        if Q[state, action] == 1:
            y = int(next_state / 8)
            x = int((next_state / 8 - y) * 8)

            goal_pos_sum = goal_pos[0] + goal_pos[1]
            distance = (goal_pos[0] - x) + (goal_pos[1] - y)

            if state == next_state or (terminated and reward == 0):
                state_chg = 0
            else:
                state_chg = 1

            Q[state, action] = (Q[state, action] * ((goal_pos_sum - distance) / goal_pos_sum)) * state_chg

        if terminated and reward == 1:
            done = True
            break

        state = next_state

        if terminated and reward == 0:
            state, _ = env.reset()

    state, _ = env.reset()

done = False
fails = 0

while not done:
    maksimum = max(Q[next_state, :])

    max_index = [i for i, x in enumerate(Q[next_state, :]) if x == maksimum]

    action = random.choice(max_index)

    next_state, reward, terminated, _, info = env.step(action)

    if terminated and reward == 1:
        done = True
        print("fails:", fails)
        print("Gratulacje! Osiągnięto cel!")
        env.close()

    if terminated and reward == 0:
        fails += 1
        state, _ = env.reset()
