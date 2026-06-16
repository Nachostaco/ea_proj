import time
import numpy as np
from matplotlib import pyplot as plt

from ant import Ant

np.set_printoptions(linewidth=600, precision=3)

# CITIES 4
CITY_X = [3, 2, 12, 7, 9, 3, 16, 11, 9, 2]
CITY_Y = [1, 4, 2, 4.5, 9, 1.5, 11, 8, 10, 7]
NUMBER_OF_CITIES = len(CITY_X) # N

NUMBER_OF_ANTS = NUMBER_OF_CITIES # m
ALPHA = 1
BETA = 5
PHEROMONE_EVAPORATION_RATE = 0.1 # rho
T_MAX = 100

TEXT_OFFSET = 0.05  # So the text does not overlap points representing cities
inv_distance_matrix = np.zeros(shape=(NUMBER_OF_CITIES, NUMBER_OF_CITIES), dtype=float)
quasi_decision_matrix = np.zeros(shape=(NUMBER_OF_CITIES, NUMBER_OF_CITIES), dtype=float)
pheromone_matrix = np.zeros(shape=(NUMBER_OF_CITIES, NUMBER_OF_CITIES), dtype=float)

def precompute_inv_distance_matrix() -> None:
    global inv_distance_matrix

    inv_distance_matrix = np.zeros((NUMBER_OF_CITIES, NUMBER_OF_CITIES))
    for i in range(NUMBER_OF_CITIES):
        for j in range(i + 1):
            distance = np.sqrt((CITY_X[i] - CITY_X[j]) ** 2 + (CITY_Y[i] - CITY_Y[j]) ** 2)
            inv_distance = 1 / distance if distance != 0 else 0
            inv_distance_matrix[i][j] = inv_distance
            inv_distance_matrix[j][i] = inv_distance

def compute_quasi_decision_matrix() -> None:
    global quasi_decision_matrix, pheromone_matrix

    for i in range(NUMBER_OF_CITIES):
        for j in range(NUMBER_OF_CITIES):
            quasi_decision_matrix[i][j] = pheromone_matrix[i][j] ** ALPHA * inv_distance_matrix[i][j] ** BETA

def compute_arc_probabilities(ant: Ant) -> np.ndarray:
    global quasi_decision_matrix

    decisions = np.zeros(shape=(1, NUMBER_OF_CITIES), dtype=float)
    arc_probs = np.zeros(shape=(1, NUMBER_OF_CITIES), dtype=float)

    unvisited_cities_mask = np.ones(decisions.shape[1], dtype=bool)
    unvisited_cities_mask[ant.visited_cities_indices] = False

    for j in range(NUMBER_OF_CITIES):
        decisions[0][j] = (quasi_decision_matrix[ant.current_city_index][j] /
                        np.sum(quasi_decision_matrix[ant.current_city_index][unvisited_cities_mask]))

    for j in range(NUMBER_OF_CITIES):
        if j in ant.visited_cities_indices:
            arc_probs[0][j] = 0
            continue

        arc_probs[0][j] = (decisions[0][j] / np.sum(decisions[0][unvisited_cities_mask]))

    return arc_probs

def compute_pheromone_deposition_matrix(ants: list[Ant]) -> np.ndarray:
    deposition_matrix = np.zeros(shape=(NUMBER_OF_CITIES, NUMBER_OF_CITIES), dtype=float)
    for k in range(NUMBER_OF_ANTS):
        ant = ants[k]
        for a in range(NUMBER_OF_CITIES):
            i = ant.visited_cities_indices[a]
            j = ant.visited_cities_indices[a + 1 if a + 1 < NUMBER_OF_CITIES else 0]
            inv_tour_length = 1 / compute_tour_length(ant)
            deposition_matrix[i][j] += inv_tour_length
            deposition_matrix[j][i] += inv_tour_length

    return deposition_matrix

def compute_tour_length(ant: Ant) -> float:
    global inv_distance_matrix

    tour_length = 0
    for i in range(NUMBER_OF_CITIES - 1):
        tour_length += inv_distance_matrix[ant.visited_cities_indices[i]][ant.visited_cities_indices[i + 1]] ** -1
    # Add the distance between the last city and the first city
    tour_length += (
            inv_distance_matrix[ant.visited_cities_indices[0]][ant.visited_cities_indices[NUMBER_OF_CITIES - 1]] ** -1)

    return tour_length

def plot_solution(solution):

    # Lines connecting cities
    for i in range(len(solution) - 1):
        plt.plot([CITY_X[solution[i]], CITY_X[solution[i + 1]]],
                 [CITY_Y[solution[i]], CITY_Y[solution[i + 1]]],
                 c='blue')
    plt.plot([CITY_X[solution[-1]], CITY_X[solution[0]]],
             [CITY_Y[solution[-1]], CITY_Y[solution[0]]],
             c='blue')

    # City points with names
    plt.scatter(CITY_X, CITY_Y, c='red')
    for i in range(len(solution)):
        plt.text(CITY_X[i] + TEXT_OFFSET, CITY_Y[i] + TEXT_OFFSET, str(i), color="black", fontsize=12)

    plt.title("Shortest route found for visiting all given cities")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def main():
    global pheromone_matrix, inv_distance_matrix

    start = time.time()

    # Initialize variables for the best solution found
    best_tour: list[int] = []
    best_tour_length = np.inf

    # Generate ants with different start cities
    start_city_indices = np.random.permutation(NUMBER_OF_CITIES).tolist()
    ants = [Ant(start_city_indices[i]) for i in range(NUMBER_OF_ANTS)]

    # Precompute a distance matrix
    precompute_inv_distance_matrix()
    # Deposit a small amount of pheromone on all arcs
    pheromone_matrix = np.full(shape=(NUMBER_OF_CITIES, NUMBER_OF_CITIES),
                               fill_value=np.min(inv_distance_matrix[inv_distance_matrix > 0]))

    for t in range(T_MAX):
        # Compute a quasi-decision matrix (used to compute elements of the decision matrix)
        compute_quasi_decision_matrix()

        for k in range(NUMBER_OF_ANTS):
            ant = ants[k]

            # Trace ant's tour
            for i in range(NUMBER_OF_CITIES - 1):
                arc_probabilities = compute_arc_probabilities(ant)
                chosen_city_index = np.random.choice(NUMBER_OF_CITIES, p=arc_probabilities[0])
                ant.move(chosen_city_index)

            tour_length = compute_tour_length(ant)
            if tour_length < best_tour_length:
                best_tour = ant.visited_cities_indices.copy()
                best_tour_length = tour_length

        # Update pheromone levels
        pheromone_matrix *= (1 - PHEROMONE_EVAPORATION_RATE)
        pheromone_matrix += compute_pheromone_deposition_matrix(ants)

        # Reset ants' memory
        for k in range(NUMBER_OF_ANTS):
            ants[k].reset()

    end = time.time()

    print(f"Optimal tour found: {best_tour}")
    print(f"Length of the optimal tour found: {best_tour_length:.4f}")
    print(f"Time taken: {end - start:.3f} s")

    plot_solution(best_tour)

if __name__ == '__main__':
    main()