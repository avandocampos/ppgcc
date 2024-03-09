# Este código executa um algoritmo genético para resolver o problema de alocação de salas de aula

import random

# Definição de dados do problema
salas = ['Sala A', 'Sala B', 'Sala C', 'Sala D', 'Sala E', 'Sala F', 'Sala G', 'Sala H', 'Sala I', 'Sala J']
professores = [
    'Prof. 1',
    'Prof. 2',
    'Prof. 3',
    'Prof. 4',
    'Prof. 5',
    'Prof. 6',
    'Prof. 7',
    'Prof. 8',
    'Prof. 9',
    'Prof. 10',
    'Prof. 11',
    'Prof. 12',
    'Prof. 13',
    'Prof. 14',
    'Prof. 15'
    ]
dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta']
horarios = ['08:00', '10:00', '14:00', '16:00', '18:00']

# Função para criar uma população inicial de horários
def create_initial_population(tamanho_populacao):
    populacao = []
    for _ in range(tamanho_populacao):
        horario_semanal = []
        for dia in dias_semana:
            aulas_dia = []
            for horario in horarios:
                aula = [random.choice(professores), random.choice(salas), horario]
                aulas_dia.append(aula)
            horario_semanal.append(aulas_dia)
        populacao.append(horario_semanal)
    return populacao

# Função de aptidão (fitness)
def calculate_fitness(horario):
    score = 100

    # Verificar conflitos de horário
    for dia in horario:
        professores_por_horario = {}
        for aula in dia:
            professor, sala, horario = aula
            if horario not in professores_por_horario:
                professores_por_horario[horario] = set()
            if professor in professores_por_horario[horario]:
                score -= 10  # Penalidade por conflito de horário
            else:
                professores_por_horario[horario].add(professor)

    # Verificar distribuição das aulas
    aulas_por_dia = [len(dia) for dia in horario]
    max_aulas = max(aulas_por_dia)
    min_aulas = min(aulas_por_dia)
    if max_aulas - min_aulas > 1:
        score -= 5  # Penalidade por distribuição desigual das aulas

    # Adicionar outros critérios aqui...

    return score

# Função de seleção por torneio
def tournament_selection(populacao, aptidoes, tamanho_torneio):
    torneio = random.sample(list(zip(populacao, aptidoes)), tamanho_torneio)
    torneio.sort(key=lambda x: x[1])
    return torneio[0][0]

# Função de crossover (recombinação)
def crossover(pai1, pai2):
    ponto_corte = random.randint(1, len(pai1) - 1)
    filho = pai1[:ponto_corte] + pai2[ponto_corte:]
    return filho

# Função de mutação
def mutation(individuo):
    indice1 = random.randint(0, len(individuo) - 1)
    indice2 = random.randint(0, len(individuo[0]) - 1)
    individuo[indice1][indice2][0] = random.choice(professores)
    individuo[indice1][indice2][1] = random.choice(salas)
    individuo[indice1][indice2][2] = random.choice(horarios)
    return individuo


if __name__ == "__main__":
    print("Algoritmo genético para alocação de salas")
    print("-----------------------------------------")
    # Parâmetros do algoritmo genético
    tamanho_populacao = 10
    tamanho_torneio = 3
    numero_geracoes = 100
    
    # Criar população inicial
    populacao = create_initial_population(tamanho_populacao)
    
    # Loop principal do algoritmo genético
    for geracao in range(numero_geracoes):
        # Calcular aptidão de cada indivíduo na população
        aptidoes = [calculate_fitness(horario) for horario in populacao]
    
        # Seleção, crossover e mutação para criar a próxima geração
        nova_populacao = []
        for _ in range(tamanho_populacao):
            pai1 = tournament_selection(populacao, aptidoes, tamanho_torneio)
            pai2 = tournament_selection(populacao, aptidoes, tamanho_torneio)
            filho = crossover(pai1, pai2)
            if random.random() < 0.1:  # Taxa de mutação de 10%
                filho = mutation(filho)
            nova_populacao.append(filho)
    
        # Atualizar a população
        populacao = nova_populacao
    
    # Encontrar e imprimir a melhor solução encontrada
    melhor_horario = min(populacao, key=calculate_fitness)
    print("Melhor horário encontrado:")
    for dia, aulas_dia in zip(dias_semana, melhor_horario):
        print(dia)
        for aula in aulas_dia:
            print(f"Professor: {aula[0]}, Sala: {aula[1]}, Horário: {aula[2]}")
    