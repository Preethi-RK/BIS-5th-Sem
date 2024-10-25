import numpy as np
import random

# Problem parameters
n_cities = 10  # Number of cities
distances = np.random.randint(10, 100, size=(n_cities, n_cities))  # Distance matrix
np.fill_diagonal(distances, 0)  # Distance from city to itself is 0

# ACO parameters
n_ants = 5       # Number of ants
n_iterations = 100
alpha = 1.0      # Influence of pheromone
beta = 2.0       # Influence of distance
evaporation_rate = 0.5
pheromone_constant = 100  # Constant for pheromone update

# Initialize pheromones
pheromone = np.ones((n_cities, n_cities))  # All edges initially have the same pheromone level

def calculate_path_length(path):
    """ Calculate the total length of the path """
    length = sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1))
    length += distances[path[-1], path[0]]  # Return to starting city
    return length

def construct_solution(pheromone):
    """ Construct a solution for each ant """
    path = [random.randint(0, n_cities - 1)]
    for _ in range(n_cities - 1):
        current_city = path[-1]
        probabilities = []
        for next_city in range(n_cities):
            if next_city not in path:
                # Calculate probability of moving to next_city
                tau = pheromone[current_city, next_city] ** alpha
                eta = (1.0 / distances[current_city, next_city]) ** beta
                probabilities.append((next_city, tau * eta))
        total = sum(prob[1] for prob in probabilities)
        probabilities = [(city, prob / total) for city, prob in probabilities]
        next_city = random.choices([city for city, _ in probabilities], 
                                   weights=[prob for _, prob in probabilities])[0]
        path.append(next_city)
    return path

def update_pheromones(pheromone, all_paths):
    """ Update pheromones on paths based on quality (length) """
    pheromone *= (1 - evaporation_rate)  # Evaporation
    for path, path_length in all_paths:
        pheromone_update = pheromone_constant / path_length
        for i in range(len(path) - 1):
            pheromone[path[i], path[i + 1]] += pheromone_update
            pheromone[path[i + 1], path[i]] += pheromone_update
        # Complete the cycle
        pheromone[path[-1], path[0]] += pheromone_update
        pheromone[path[0], path[-1]] += pheromone_update

# Run ACO
best_path = None
best_length = float('inf')
for iteration in range(n_iterations):
    all_paths = []
    for ant in range(n_ants):
        path = construct_solution(pheromone)
        path_length = calculate_path_length(path)
        all_paths.append((path, path_length))
        # Track best solution
        if path_length < best_length:
            best_path, best_length = path, path_length
            print(f"Iteration {iteration}, new best length: {best_length}")
    # Update pheromones based on all paths
    update_pheromones(pheromone, all_paths)

print("\nBest path found:", best_path)
print("Best path length:", best_length)
