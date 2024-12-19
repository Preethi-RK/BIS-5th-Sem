import numpy as np

# Objective function (example: simple mathematical function)
def objective_function(x):
    return np.sum(x**2)

# Function to initialize population (random programs)
def initialize_population(pop_size, dim, lb, ub):
    population = []
    for _ in range(pop_size):
        individual = np.random.uniform(lb, ub, dim)
        population.append(individual)
    return np.array(population)

# Function to evaluate fitness of the population
def evaluate_fitness(population, objective_function):
    fitness = np.apply_along_axis(objective_function, 1, population)
    return fitness

# Genetic operators: selection, crossover, mutation
def selection(population, fitness):
    sorted_idx = np.argsort(fitness)
    return population[sorted_idx[:len(population)//2]]  # Select best half

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
def gene_expression_programming(n, dim, lb, ub, Max_iter, objective_function):
    population = initialize_population(n, dim, lb, ub)
    fitness = evaluate_fitness(population, objective_function)
    
    for _ in range(Max_iter):
        selected_population = selection(population, fitness)
        next_generation = []
        
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutation(child1, lb, ub))
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
    Max_iter = 100   # Maximum iterations

    best_solution, best_fitness = gene_expression_programming(n, dim, lb, ub, Max_iter, objective_function)
    
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
