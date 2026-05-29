import math
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# More readable printing of large NumPy arrays
np.set_printoptions(linewidth=600)

# CITIES 3
# city_X = [0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5]
# city_Y = [1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]
# number_of_cities = len(city_X) # N

# CITIES 4
# city_X = [3, 2, 12, 7, 9, 3, 16, 11, 9, 2]
# city_Y = [1, 4, 2, 4.5, 9, 1.5, 11, 8, 10, 7]
# number_of_cities = len(city_X) # N

# VOIVODESHIP CITIES (coordinates are loaded in runtime)
city_X = []
city_Y = []
number_of_cities = 16 # N
city_names = ["Wrocław", "Bydgoszcz", "Lublin", "Gorzów Wielkopolski",
              "Łódź", "Kraków", "Warszawa", "Opole",
              "Rzeszów", "Białystok", "Gdańsk", "Katowice",
              "Kielce", "Olsztyn", "Poznań", "Szczecin"]

population_size = 500 # P
selection_proportion = 0.9  # n
mutation_probability = 0.25  # p_m
maximum_generations = 1000  # T_max

pop_sizes = [100, 300, 500]
sel_props = [0.5, 0.7, 0.9]
mut_probs = [0.1, 0.3, 0.5]
num_of_configs = len(pop_sizes) * len(sel_props) * len(mut_probs)
trials = 10

empty_allele_value = -1  # Representation of empty allele
text_offset = 0.05  # So the text does not overlap points representing cities

inv_distance_matrix = np.zeros(shape=(number_of_cities, number_of_cities), dtype=float)

individuals_fitness = []
total_fitness = []


def precompute_distance_matrix():
    global inv_distance_matrix

    distance_matrix = np.zeros((number_of_cities, number_of_cities))
    for i in range(number_of_cities):
        for j in range(i + 1):
            distance = np.sqrt((city_X[i] - city_X[j]) ** 2 + (city_Y[i] - city_Y[j]) ** 2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

# Used for voivodeship cities
def load_cities_data_from_file():
    global inv_distance_matrix, city_X, city_Y
    data = pd.read_csv("../../resources/voivodeship_cities.csv")
    city_X = data["X"].to_list()
    city_Y = data["Y"].to_list()
    data = data.drop(columns=data.columns[0:5])
    distance_matrix = data.to_numpy()

def generate_population(size: int):
    return np.array([np.random.permutation(number_of_cities) for _ in range(size)])


def get_total_distance(city_indices):
    # Precompute distance matrix
    travel_distance = 0
    for k in range(number_of_cities - 1):
        travel_distance += inv_distance_matrix[city_indices[k]][city_indices[k + 1]]

    # Add the distance between the last city and the first city
    travel_distance += inv_distance_matrix[city_indices[0]][city_indices[number_of_cities - 1]]

    return travel_distance


def precompute_fitness_values(population):
    global individuals_fitness, total_fitness

    cost_values = [get_total_distance(individual) for individual in population]  # f_i
    max_cost = max(cost_values)  # m_f

    individuals_fitness = [max_cost - cost for cost in cost_values]  # t_i
    total_fitness = sum(individuals_fitness)  # t_s


def roulette_selection(population):
    """
    Returns a random individual from the given population based on the fitness values.
    :param population:
    :return: A randomly selected individual from the given population.
    """
    global individuals_fitness, total_fitness

    random_number = np.random.uniform(0, total_fitness)  # r - randomized from [0; t_s), not [0; t_s], but who cares

    cumulative_sum = 0
    for individual, fitness_value in zip(population, individuals_fitness):
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

    # Lines connecting cities
    for i in range(len(solution) - 1):
        plt.plot([city_X[solution[i]], city_X[solution[i + 1]]],
                 [city_Y[solution[i]], city_Y[solution[i + 1]]],
                 c='blue')
    plt.plot([city_X[solution[-1]], city_X[solution[0]]],
             [city_Y[solution[-1]], city_Y[solution[0]]],
             c='blue')

    # City points with names
    plt.scatter(city_X, city_Y, c='red')
    for i in range(len(solution)):
        plt.text(city_X[i] + text_offset, city_Y[i] + text_offset, city_names[i], color="black", fontsize=12)

    plt.title("Shortest found route for visiting all voivodeship cities")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


# def main():
#     global individuals_fitness, total_fitness
#
#     # Precompute distance matrix to avoid repeatedly calculating distances
#     precompute_distance_matrix()
#
#     start_time = time.time()
#
#     params_idx = 1
#
#     for pop_size in pop_sizes:
#         for sel_prop in sel_props:
#             for mut_prob in mut_probs:
#
#                 minimal_dist_sum = 0
#
#                 for _ in range(trials):
#                     # Generate initial population
#                     population = generate_population(pop_size)
#
#                     for generation in range(maximum_generations):
#                         new_population = []
#
#                         precompute_fitness_values(population)
#
#                         # Select n*P individuals for crossover
#                         crossover_population = [roulette_selection(population) for _ in
#                                                 range(math.ceil(sel_prop * pop_size))]
#
#                         # Create n*P offspring
#                         while len(new_population) < sel_prop * pop_size:
#                             i_p1, i_p2 = np.random.choice(len(crossover_population), 2)
#                             p1, p2 = crossover_population[i_p1], crossover_population[i_p2]
#                             o1, o2 = crossover(p1, p2)
#                             new_population.append(o1)
#                             new_population.append(o2)
#
#                         new_population = np.array(new_population)
#
#                         # Mutate randomly selected offspring
#                         for idx in range(len(new_population)):
#                             if np.random.rand() >= mut_prob:
#                                 new_population[idx] = mutate(new_population[idx])
#
#                         # Replace the current population with the new population
#                         combined_population = np.concatenate((population, new_population))
#                         combined_population = np.array(sorted(combined_population, key=get_total_distance))
#                         population = combined_population[:pop_size]
#
#                     minimal_dist_sum += get_total_distance(population[0])
#
#                 print(f"{params_idx}/{num_of_configs} - Population size: {pop_size}, Selection proportion: {sel_prop},"
#                       f" Mutation probability: {mut_prob},")
#                 print(f"Minimal mean distance: {minimal_dist_sum / trials:.4f}")
#
#                 params_idx += 1
#
#     end_time = time.time()
#     print(f"Time taken in total: {end_time - start_time} seconds")

def main():
    global individuals_fitness, total_fitness

    # Precompute distance matrix to avoid repeatedly calculating distances
    # precompute_distance_matrix()

    # Load in the distance matrix for voivodeship cities
    load_cities_data_from_file()

    print(get_total_distance(np.array([5, 8, 2, 12, 4, 6, 9, 13, 10, 1, 14, 15, 3, 0, 7, 11])))

    start_time = time.time()

    # Generate initial population
    population = generate_population(population_size)

    for generation in range(maximum_generations):
        new_population = []

        precompute_fitness_values(population)

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
        combined_population = np.array(sorted(combined_population, key=get_total_distance))
        population = combined_population[:population_size]

        if generation % 50 == 0:
            print(f"Generation {generation}: Best solution: {population[0]}, Total distance (cost): {get_total_distance(population[0]):.2f} km")

    end_time = time.time()

    print(f"Best solution: {population[0]}")
    print(f"Total distance (cost): {get_total_distance(population[0]):.2f} km")
    print(f"Time taken: {end_time - start_time:.2f} s")

    plot_solution(population[0])


if __name__ == "__main__":
    main()
