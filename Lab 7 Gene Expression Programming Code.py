# Objective function (example: simple mathematical function)
import numpy as np

def objective_function(x):
    return np.sum(x**2)

# Function to initialize population (random programs)
def initialize_population(pop_size, dim, lb, ub):
    return np.random.uniform(lb, ub, (pop_size, dim))

# Function to evaluate fitness of the population
def evaluate_fitness(population, objective_function):
    return np.apply_along_axis(objective_function, 1, population)

# Genetic operators: selection, crossover, mutation
def selection(population, fitness, num_selected):
    sorted_idx = np.argsort(fitness)
    return population[sorted_idx[:num_selected]]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutation(child, lb, ub):
    mutation_point = np.random.randint(0, len(child))
    child[mutation_point] = np.random.uniform(lb, ub)
    return child

# Gene Expression Programming (GEP)
def gene_expression_programming(n, dim, lb, ub, max_iter, objective_function):
    population = initialize_population(n, dim, lb, ub)
    fitness = evaluate_fitness(population, objective_function)
    
    for _ in range(max_iter):
        # Select the top 50% of the population
        num_selected = max(2, len(population) // 2)  # Ensure at least 2 individuals
        selected_population = selection(population, fitness, num_selected)
        next_generation = []

        # Reproduce and mutate to maintain population size
        while len(next_generation) < n:
            parent1, parent2 = selected_population[np.random.choice(len(selected_population), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutation(child1, lb, ub))
            if len(next_generation) < n:  # Ensure population size
                next_generation.append(mutation(child2, lb, ub))

        population = np.array(next_generation)
        fitness = evaluate_fitness(population, objective_function)
    
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

# Example usage
if __name__ == "__main__":
    n = 30           # Population size
    dim = 10         # Number of dimensions
    lb = -5          # Lower bound
    ub = 5           # Upper bound
    max_iter = 100   # Maximum iterations

    best_solution, best_fitness = gene_expression_programming(n, dim, lb, ub, max_iter, objective_function)
    
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

