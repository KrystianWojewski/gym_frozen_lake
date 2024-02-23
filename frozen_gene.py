import random

import gym
import numpy as np

env = gym.make('FrozenLake8x8-v1', render_mode='ansi', is_slippery=False)

def initialize_population(population_size, gene_length):
    population = []
    for _ in range(population_size):
        individual = np.random.choice(env.action_space.n, size=gene_length)
        population.append(individual)
    return population

def evaluate_fitness(individual):
    env.reset()
    total_reward = 0
    for action in individual:
        _, reward, terminated, _, _ = env.step(action)
        total_reward += 1
        if terminated and reward == 0:
            total_reward -= 100
            return total_reward
        elif terminated and reward == 1:
            total_reward -= 300
            return abs(total_reward)
    return total_reward

def tournament_selection(population, fitness_scores, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        selected_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_parents.append(population[selected_index])
    return selected_parents


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, env.action_space.n - 1)
    return individual

def genetic_algorithm(population, num_generations, tournament_size, mutation_rate):
    for generation in range(num_generations):
        fitness_scores = [evaluate_fitness(individual) for individual in population]

        # Wybór rodziców
        parents = tournament_selection(population, fitness_scores, tournament_size)

        # Krzyżowanie
        offspring = []
        for i in range(0, population_size, 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            offspring.append(child1)
            offspring.append(child2)

        # Mutacja
        mutated_offspring = [mutate(child, mutation_rate) for child in offspring]

        # Zastąpienie populacji potomnej populacją rodzicielską
        population = mutated_offspring

    fitness_scores = [evaluate_fitness(individual) for individual in population]

    best_individual_index = np.argmax(fitness_scores)
    best_individual = population[best_individual_index]
    best_fitness = fitness_scores[best_individual_index]

    return best_individual, best_fitness

population_size = 100
gene_length = 100
population = initialize_population(population_size, gene_length)

num_generations = 100
tournament_size = 5
mutation_rate = 0.1

best_individual, best_fitness = genetic_algorithm(population, num_generations, tournament_size, mutation_rate)
print("Najlepszy osobnik:", best_individual)
print("Najlepsza wartość funkcji celu:", best_fitness)

env.reset()
for i in best_individual:
    _, reward, terminated, _, _ = env.step(i)
print(terminated)
print(reward)
