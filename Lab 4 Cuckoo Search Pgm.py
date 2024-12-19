import numpy as np

# Objective function (to minimize)
def objective_function(x):
    return np.sum(x**2)  # Example: Sphere function

# Lévy flight
def levy_flight(beta):
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v)**(1 / beta)
    return step[0]

# Cuckoo Search Algorithm
def cuckoo_search(n=25, pa=0.25, max_iter=100, dim=5, bounds=(-10, 10)):
    # Initialize population
    lower, upper = bounds
    nests = np.random.uniform(lower, upper, (n, dim))
    fitness = np.apply_along_axis(objective_function, 1, nests)
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx]
    best_fitness = fitness[best_idx]

    for t in range(max_iter):
        # Generate new solutions using Lévy flight
        for i in range(n):
            new_solution = nests[i] + levy_flight(1.5) * np.random.randn(dim)
            new_solution = np.clip(new_solution, lower, upper)  # Enforce boundaries
            new_fitness = objective_function(new_solution)
            
            # Replace if new solution is better
            if new_fitness < fitness[i]:
                nests[i] = new_solution
                fitness[i] = new_fitness

        # Abandon a fraction of the worst nests
        sorted_indices = np.argsort(fitness)
        num_abandon = int(pa * n)
        worst_indices = sorted_indices[-num_abandon:]

        for idx in worst_indices:
            nests[idx] = np.random.uniform(lower, upper, dim)
            fitness[idx] = objective_function(nests[idx])

        # Update the best solution
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_nest = nests[best_idx]
            best_fitness = fitness[best_idx]

        print(f"Iteration {t + 1}: Best Fitness = {best_fitness}")

    return best_nest, best_fitness

# Run the Cuckoo Search Algorithm
best_solution, best_fitness = cuckoo_search(n=25, pa=0.25, max_iter=100, dim=5, bounds=(-10, 10))
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
