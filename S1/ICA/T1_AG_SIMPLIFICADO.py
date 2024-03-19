import random

# Parâmetros do problema
NUM_DISCIPLINAS = 15
NUM_OFERTADAS = 10  # Número de disciplinas a serem ofertadas
PESOS_DISCIPLINAS = [3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Pesos das disciplinas
POP_SIZE = 100
GEN_SIZE = 50
MUTATION_RATE = 0.1

# Dicionário para vincular o número da disciplina ao seu identificador
disciplina_id = {i: f"Disciplina {i+1}" for i in range(NUM_DISCIPLINAS)}

# Função para inicializar a população aleatoriamente respeitando o número de disciplinas ofertadas
def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        individuo = random.sample(range(NUM_DISCIPLINAS), NUM_OFERTADAS)
        population.append(individuo)
    return population

# Função para calcular o fitness de um indivíduo
def calculate_fitness(individual):
    total_peso = sum(PESOS_DISCIPLINAS[disciplina] for disciplina in individual)
    return total_peso

# Função para realizar a seleção de pais por torneio
def select_parents(population):
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, 5)
        winner = max(tournament, key=calculate_fitness)
        parents.append(winner)
    return parents

# Função para realizar o crossover entre dois pais
def crossover(parent1, parent2):
    point = random.randint(1, NUM_OFERTADAS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Função para realizar a mutação de um indivíduo
def mutate(individual):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, NUM_OFERTADAS - 1)
        individual[index] = random.randint(0, NUM_DISCIPLINAS - 1)

# Função principal do algoritmo genético
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

    print("Melhor solução encontrada:")
    for disciplina in best_solution:
        print(f"{disciplina_id[disciplina]} - Peso: {PESOS_DISCIPLINAS[disciplina]}")
    print("Fitness da melhor solução:", best_fitness)

if __name__ == "__main__":
    genetic_algorithm()