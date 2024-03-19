import random

# Problem parameters
NUM_SUBJECTS = 11
NUM_OFFERED = 6  # Number of subjects to be offered
SUBJECT_WEIGHTS = [5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1]  # Subject weights
POP_SIZE = 10
GEN_SIZE = 50
MUTATION_RATE = 0.1

# Dictionary to map subject numbers to their identifiers
subject_id = {
    1: "TC",
    2: "SIG",
    3: "SD",
    4: "RA",
    5: "PI",
    6: "PAD",
    7: "MSH",
    8: "ICA",
    9: "ASE",
    10: "AP",
    11: "AM"
}

# Function to initialize the population randomly respecting the number of offered subjects
def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        individual = random.sample(range(NUM_SUBJECTS), NUM_OFFERED)
        population.append(individual)
    return population

# Function to calculate the fitness of an individual
def calculate_fitness(individual):
    total_weight = sum(SUBJECT_WEIGHTS[subject] for subject in individual)
    return total_weight

# Function to perform parent selection by tournament
def select_parents(population):
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, 5)
        winner = max(tournament, key=calculate_fitness)
        parents.append(winner)
    return parents

# Function to perform crossover between two parents
def crossover(parent1, parent2):
    point = random.randint(1, NUM_OFFERED - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Function to perform mutation of an individual
def mutate(individual):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, NUM_OFFERED - 1)
        individual[index] = random.randint(0, NUM_SUBJECTS - 1)

# Main genetic algorithm function
def genetic_algorithm():
    population = initialize_population()

    for _ in range(GEN_SIZE):
        parents = select_parents(population)
        offspring = []

        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            offspring.extend([child1, child2])

        population = offspring

    best_solution = max(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_solution)

    print("Best solution found:")
    for subject in best_solution:
        print(f"{subject_id[subject]} - Weight: {SUBJECT_WEIGHTS[subject]}")
    print("Fitness of the best solution:", best_fitness)

if __name__ == "__main__":
    genetic_algorithm()
