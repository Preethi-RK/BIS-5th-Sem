import numpy as np

# Objective function to optimize (example: Sphere function)
def objective_function(x):
    return np.sum(x**2)

# Update the position of wolves
def update_position(X, alpha, beta, delta, A, B, C, lb, ub):
    D_alpha = np.abs(C[0] * alpha - X)
    D_beta = np.abs(C[1] * beta - X)
    D_delta = np.abs(C[2] * delta - X)
    
    X_new = X + A[0] * D_alpha + B[0] * D_beta + C[0] * D_delta
    # Apply bounds
    X_new = np.clip(X_new, lb, ub)
    return X_new

# Grey Wolf Optimizer (GWO)
def gwo(n, dim, lb, ub, Max_iter, objective_function):
    # Initialize population of wolves
    wolves = np.random.uniform(lb, ub, (n, dim))
    fitness = np.apply_along_axis(objective_function, 1, wolves)
    
    # Sort wolves based on fitness
    sorted_indices = np.argsort(fitness)
    alpha = wolves[sorted_indices[0]]  # Best solution
    beta = wolves[sorted_indices[1]]   # Second best
    delta = wolves[sorted_indices[2]]  # Third best
    
    # Main loop
    for t in range(Max_iter):
        # Update coefficients A, B, C
        A = 2 * np.random.rand(3, dim) - 1
        B = 2 * np.random.rand(3, dim) - 1
        C = 2 * np.random.rand(3, dim) - 1
        
        # Update position of each wolf
        for i in range(n):
            wolves[i] = update_position(wolves[i], alpha, beta, delta, A, B, C, lb, ub)
        
        # Evaluate fitness
        fitness = np.apply_along_axis(objective_function, 1, wolves)
        
        # Update alpha, beta, delta wolves
        sorted_indices = np.argsort(fitness)
        alpha = wolves[sorted_indices[0]]
        beta = wolves[sorted_indices[1]]
        delta = wolves[sorted_indices[2]]
    
    return alpha, fitness[sorted_indices[0]]  # Return best solution

# Example usage
if __name__ == "__main__":
    n = 30               # Population size (number of wolves)
    dim = 10             # Number of dimensions
    lb = -5              # Lower bound of the search space
    ub = 5               # Upper bound of the search space
    Max_iter = 100       # Maximum number of iterations

    # Run the Grey Wolf Optimizer
    best_solution, best_fitness = gwo(n, dim, lb, ub, Max_iter, objective_function)
    
    # Output the results
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
