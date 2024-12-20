import numpy as np

# Define the optimization function
def fitness_function(x):
    return x**2

# Initialize parameters
num_cells = 10
grid_size = 1.0
iterations = 100
neighborhood_size = 1

# Initialize population
cells = np.random.uniform(-grid_size, grid_size, num_cells)

# Main loop
for _ in range(iterations):
    # Evaluate fitness
    fitness = np.array([fitness_function(cell) for cell in cells])

    # Update states
    new_cells = np.copy(cells)
    for i in range(num_cells):
        # Get neighbors
        neighbors = cells[max(0, i-neighborhood_size):min(num_cells, i+neighborhood_size+1)]
        # Update cell based on neighbors
        new_cells[i] = np.mean(neighbors) + np.random.uniform(-0.1, 0.1)  # Add some noise

    cells = new_cells

# Output the best solution
best_cell = cells[np.argmin(fitness)]
print(f"Best solution found: {best_cell}")
print(f"Fitness: {fitness_function(best_cell)}")
