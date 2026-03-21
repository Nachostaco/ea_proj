import math
import time

import numpy as np
from matplotlib import pyplot as plt

# CITIES
city_X = [0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5]
city_Y = [1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]
number_of_cities = 10 # N

population_size = 250 # P
selection_proportion = 0.8 # n
mutation_probability = 0.2 # p_m
maximum_generations = 1000 # T_max

empty_allele_value = -1 # Representation of empty allele
text_offset = 0.1 # So the text does not overlap points representing cities

dist_matrix = []

def generate_population(size: int):
    return np.array([np.random.permutation(number_of_cities) for _ in range(size)])

def get_total_distance(city_indices):
    travel_distance = 0
    for k in range(number_of_cities - 1):
        x_diff = (city_X[city_indices[k]] - city_X[city_indices[k + 1]])
        y_diff = (city_Y[city_indices[k]] - city_Y[city_indices[k + 1]])
        travel_distance += np.sqrt(x_diff**2 + y_diff**2)

    # Add the distance between the last city and the first city
    travel_distance += np.sqrt(
        (city_X[city_indices[0]] - city_X[city_indices[number_of_cities - 1]]) ** 2 +
        (city_Y[city_indices[0]] - city_Y[city_indices[number_of_cities - 1]]) ** 2
    )

    return travel_distance

def roulette_selection(population):
    """
    Returns a random individual from the given population based on the fitness values.
    :param population:
    :return: A randomly selected individual from the given population.
    """
    cost_values = [get_total_distance(individual) for individual in population] # f_i
    max_cost = max(cost_values) # m_f

    fitness = [max_cost - cost for cost in cost_values] # t_i
    total_fitness = sum(fitness) # t_s

    random_number = np.random.uniform(0, total_fitness) # r - randomized from [0; t_s), not [0; t_s], but who cares

    cumulative_sum = 0
    for individual, fitness_value in zip(population, fitness):
        cumulative_sum += fitness_value
        if cumulative_sum >= random_number:
            return individual

    return np.random.choice(population)

def _crossover_fill_remaining(o, p):
    for i in range(len(o)):
        if o[i] == empty_allele_value:
            o[i] = p[i]
    return o

def _create_offspring(p1, p2):
    o = [empty_allele_value] * number_of_cities
    o[0] = p1[0]
    prev_index = 0

    for _ in range(1, number_of_cities):
        if p2[prev_index] in o:
            return _crossover_fill_remaining(o, p2)
        new_index = np.where(p1 == p2[prev_index])[0][0]
        o[new_index] = p2[prev_index]
        prev_index = new_index

    return o

def crossover(p1, p2):
    """
    Creates two offspring from two parents.
    :param p1:
    :param p2:
    :return: Two offspring in a tuple.
    """
    o1 = _create_offspring(p1, p2)
    o2 = _create_offspring(p2, p1)
    return o1, o2

def mutate(individual):
    """
    Mutate the individual by swapping two random alleles.
    :param individual:
    :return: The individual with two random alleles swapped.
    """
    a1, a2 = np.random.choice(range(len(individual)), 2)
    individual[a1], individual[a2] = individual[a2], individual[a1]
    return individual

def plot_solution(solution):
    # City points
    plt.scatter(city_X, city_Y, c = 'red')
    for i in range(len(solution)):
        plt.text(city_X[i] + text_offset, city_Y[i] + text_offset, str(i), color = "black", fontsize = 12)

    # Lines connecting cities
    for i in range(len(solution) - 1):
        plt.plot([city_X[solution[i]], city_X[solution[i + 1]]],
                 [city_Y[solution[i]], city_Y[solution[i + 1]]],
                 c = 'blue')
    plt.plot([city_X[solution[-1]], city_X[solution[0]]],
             [city_Y[solution[-1]], city_Y[solution[0]]],
             c = 'blue')

    plt.title("Optimal found TSP route")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def main():
    # So far, the best solution found is:
    # [4, 5, 3, 2, 1, 0, 7, 9, 8, 6]
    # with total distance (cost) equal to 61.13744551656403

    start_time = time.time()

    # Generate initial population
    population = generate_population(population_size)

    for generation in range(maximum_generations):
        new_population = []

        # Select n*P individuals for crossover
        crossover_population = [roulette_selection(population) for _ in
                                range(math.ceil(selection_proportion * population_size))]

        # Create n*P offspring
        while len(new_population) < selection_proportion * population_size:
            i_p1, i_p2 = np.random.choice(len(crossover_population), 2)
            p1, p2 = crossover_population[i_p1], crossover_population[i_p2]
            o1, o2 = crossover(p1, p2)
            new_population.append(o1)
            new_population.append(o2)

        new_population = np.array(new_population)

        # Mutate randomly selected offspring
        for idx in range(len(new_population)):
            if np.random.rand() >= mutation_probability:
                new_population[idx] = mutate(new_population[idx])

        # Replace the current population with the new population
        combined_population = np.concatenate((population, new_population))
        combined_population = np.array(sorted(combined_population, key = get_total_distance))
        population = combined_population[:population_size]

        if generation % 50 == 0:
            print(f"Generation {generation}: Best solution: {population[0]}, Cost: {get_total_distance(population[0])}")

    end_time = time.time()

    print(f"Best solution: {population[0]}")
    print(f"Cost: {get_total_distance(population[0])}")
    print(f"Time taken: {end_time - start_time} seconds")

    plot_solution(population[0])

if __name__ == "__main__":
    main()
