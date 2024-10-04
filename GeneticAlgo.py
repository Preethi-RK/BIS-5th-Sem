from random import randint, random

# Objective function: maximize the number of 1s in the bitstring
def objective(bitstring):
    return -sum(bitstring)  # Return negative to minimize the negative count

# Tournament selection
def selection(pop, scores, k=3):
    # First random selection
    selection_ix = randint(0, len(pop) - 1)
    for ix in [randint(0, len(pop) - 1) for _ in range(k - 1)]:
        # Check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:  # Minimize scores
            selection_ix = ix
    return pop[selection_ix]

# Crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # Children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # Check for recombination
    if random() < r_cross:
        # Select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # Perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# Mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # Check for a mutation
        if random() < r_mut:
            # Flip the bit
            bitstring[i] = 1 - bitstring[i]

# Genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    # Initial population of random bitstrings
    pop = [[randint(0, 1) for _ in range(n_bits)] for _ in range(n_pop)]
    # Keep track of best solution
    best, best_eval = 0, objective(pop[0])
    # Enumerate generations
    for gen in range(n_iter):
        # Evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # Check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:  # Minimize scores
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        # Select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # Create the next generation
        children = []
        for i in range(0, n_pop, 2):
            # Get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # Crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # Mutation
                mutation(c, r_mut)
                # Store for next generation
                children.append(c)
        # Replace population
        pop = children
    return [best, best_eval]

# Example usage
if __name__ == "__main__":
    n_bits = 20  # Length of bitstring
    n_iter = 100  # Number of generations
    n_pop = 100  # Population size
    r_cross = 0.9  # Crossover rate
    r_mut = 0.1  # Mutation rate

    best, best_eval = genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut)
    print("Best solution: %s, Best evaluation: %.3f" % (best, best_eval))
