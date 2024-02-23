from math import sqrt

import numpy as np
import gym

def q_learning(env, goal_pos, learning_rate=0.8, discount_factor=0.95, epsilon=0.5, num_epochs=1000):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    goal_y = goal_pos[0]
    goal_x = goal_pos[1]
    max_distance = goal_y + goal_x

    Q = np.zeros((num_states, num_actions))

    for epoch in range(num_epochs):
        state, info = env.reset()
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ , info = env.step(action)

            agent_y = int(next_state / int(sqrt(env.observation_space.n+1)))
            agent_x = int((next_state / int(sqrt(env.observation_space.n+1)) - agent_y) * int(sqrt(env.observation_space.n+1)))

            distance = goal_y - agent_y + goal_x - agent_x

            if (done == False and state != next_state):
                reward = (max_distance - distance) / max_distance
            if (done == True and reward == 1):
                reward = 20

            Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

    optimal_policy = np.argmax(Q, axis=1)

    return Q, optimal_policy


env = gym.make("FrozenLake8x8-v1", render_mode="ansi", is_slippery=False)

y = int(env.observation_space.n / sqrt(env.observation_space.n+1))
x = int((env.observation_space.n / sqrt(env.observation_space.n+1) - y) * sqrt(env.observation_space.n+1))

goal_pos = x, y

Q_values, optimal_policy = q_learning(env, goal_pos)

print("Estymowane wartości Q:")
print(Q_values)

print("Optymalna polityka:")
print(optimal_policy)

env = gym.make("FrozenLake8x8-v1", render_mode="human", is_slippery=False)

state, info = env.reset()

done = False

while not done:
    state, _, done, _, _ = env.step(optimal_policy[state])
