from pulp import *
import numpy as np
import math
import copy

# Function: initialize the random permutation of the indexes of the cities
def init_pop(city_num,size=10000):
	pop = []
	for i in range(size):
		ind = np.random.permutation(city_num)
		pop.append(ind)
	return pop

# Function: calculate the distance between all cities under one specific order,
# note that the distance from the terminal city to the initial city should be included, but not within the iteration.
def evaluation_one(data,ind):
	dist = 0
	for i in range(len(ind)-1):
		dist += math.sqrt((data[ind[i+1]][0]-data[ind[i]][0])**2 + (data[ind[i+1]][1]-data[ind[i]][1])**2)
	dist += math.sqrt((data[ind[-1]][0]-data[ind[0]][0])**2 + (data[ind[-1]][1]-data[ind[0]][1])**2)
	return dist

# Function: calculate the whole distance sets w.r.t. the init_pop
def eva_all(data,pop):
	for i in range(len(pop)):
		pop.append(evaluation_one(data,pop[i]))

# Function: time to generate the new offsprings, note that symmetric splitting should be avoided, better do it randomly
def crossover(ind1,ind2):
	random_cut_point = np.random.randint(len(ind1))
	offspring1 = ind1[:random_cut_point]
	for otherelement in ind2:
		if otherelement not in offspring1:
			offspring1.append(otherelement)
	offspring2 = ind2[:random_cut_point]
	for otherelement in ind1:
		if otherelement not in offspring2:
			offspring2.append(otherelement)

	return(offspring1,offspring2)

# Function: swap two orders in one sequence to mimic the mutation progress in the natural evolution
def mutation(ind):
	mutation_point0 = np.random.randint(len(ind))
	mutation_point1 = np.random.randint(len(ind))
	ind[mutation_point0],ind[mutation_point1] = ind[mutation_point1], ind[mutation_point0]

def tournament_select(pop, k=5):
	random_select = sorted(np.random.choice(a=np.arange(len(pop)), size=k, replace=False), reverse=True)
	tournament_pop = []
	for idx in random_select:
		tournament_pop.append(pop.pop(idx))

	tournament_pop.sort(key=lambda x:x[1])
	return tournament_pop[0]

# Function: due to the limited computational capacity, the total amount of population must be kept in a certain size
#def merge_pop(init_pop,offsprings,mutations,pop_size=10000):
def merge_pop(current_pop, next_pop, pop_size, elite_ratio=0.9):
	current_pop.extend(next_pop)
	current_pop.sort(key=lambda x: x[1])
	elite_size = int(pop_size * elite_ratio)
	new_pop = current_pop[:elite_size] + current_pop[-(pop_size - elite_size):]
	return new_pop

def ga():
	# Input data
	data = parse_tsp_list("dj38.tsp")
	city_nb = len(data)

	# Parameters
	P_C = 1
	P_M = 0.3
	MAX_ITER = 10000
	MAX_NOCHANGE = 1000
	POP_SIZE = 100

	# Init pop
	rand_pop = init_population(city_nb)
	eval_pop(data, rand_pop)
	rand_pop.sort(key=lambda x: x[1])
	current_pop = rand_pop[:100]
	current_best = current_pop[0][1]

	# Evolution
	i = 0
	nochange_iter = 0
	while i <= MAX_ITER and nochange_iter <= MAX_NOCHANGE:
		print("Iteration", i, "Best", current_best)
		select_pop = copy.deepcopy(current_pop)
		next_pop = []
		while select_pop != []:
			parent1 = tournament_select(select_pop)
			parent2 = tournament_select(select_pop)
			if np.random.rand() <= P_C:
				offsprings = crossover(parent1[0], parent2[0])

				for child in offsprings:
					if np.random.rand() <= P_M:
						mutation(child)

				next_pop.extend(offsprings)

		eval_pop(data, next_pop)
		current_pop = merge_pop(current_pop, next_pop, POP_SIZE)
		if current_pop[0][1] >= current_best:
			nochange_iter += 1
		else:
			current_best = current_pop[0][1]
			nochange_iter = 0
		i += 1

	return current_pop[0]

best_sol = ga()
print(best_sol)