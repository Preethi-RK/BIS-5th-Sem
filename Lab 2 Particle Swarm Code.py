import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], size=len(bounds))
        self.velocity = np.random.uniform(-1, 1, size=len(bounds))
        self.best_position = self.position.copy()
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1 = np.random.random(size=len(self.position))
        r2 = np.random.random(size=len(self.position))
        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.best_position - self.position) +
                         c2 * r2 * (global_best_position - self.position))

    def update_position(self, bounds):
        self.position += self.velocity
        # Ensure the particle stays within the bounds
        self.position = np.clip(self.position, bounds[0], bounds[1])

def objective_function(x):
    # Example objective function: Sphere Function
    return sum(x**2)

def particle_swarm_optimization(objective_func, bounds, num_particles=30, max_iter=100):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = np.random.uniform(bounds[0], bounds[1], size=len(bounds))
    global_best_value = float('inf')

    for iteration in range(max_iter):
        for particle in particles:
            value = objective_func(particle.position)

            # Update personal best
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position.copy()

            # Update global best
            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position(bounds)

    return global_best_position, global_best_value

# Example usage
bounds = [-10, 10]  # Search space bounds
best_position, best_value = particle_swarm_optimization(objective_function, bounds)

print("Best Position:", best_position)
print("Best Value:", best_value)
