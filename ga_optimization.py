import timeit

from deap import base
from deap import creator
from deap import tools


import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

import bilstm
import elitism

# boundaries for all parameters:
# 'hidden_layer_sizes': first four values
# 'activation': ['tanh', 'relu', 'sigmoid'] -> 0, 1, 2
# 'optimizer': ['Adam', 'SGD', 'RMSprop'] -> 0, 1, 2
# 'learning_rate': float in the range of [0.0001, 0.1],
# 'batch_size': float in the range of [20, '32]

# BOUNDS_LOW = [32,   -5, -10, -20, 0,     0,     0.0001,  20]
# BOUNDS_HIGH = [128, 64,  32,  16, 2.999, 2.999, 0.1,     32]
# BOUNDS_LOW =  [ 5,  -5, -10, -20, 0,     0,     0.0001, 0    ]
# BOUNDS_HIGH = [15,  10,  10,  10, 2.999, 2.999, 2.0,    2.999]

BOUNDS_LOW =  [16,  8, -4, -8, 0,     0,     0.0001, 20]
BOUNDS_HIGH = [32, 24, 16, 16, 2.999, 2.999, 0.01,   40]


NUM_OF_PARAMS = len(BOUNDS_HIGH)

# Genetic Algorithm constants:
POPULATION_SIZE = 100
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 10  # 5
HALL_OF_FAME_SIZE = 3  # 3
CROWDING_FACTOR = 10.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 1420
random.seed(RANDOM_SEED)

# create the classifier accuracy test class:
hyperparameter_test = bilstm.BiLSTM()

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMan", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMan)

# define the layer size attributes individually:
for i in range(NUM_OF_PARAMS):
    # "attribute_0", "attribute_1", ...
    toolbox.register("attribute_" + str(i),
                     random.uniform,
                     BOUNDS_LOW[i],
                     BOUNDS_HIGH[i])

# create a tuple containing an attribute generator for each param searched:
attributes = ()
for i in range(NUM_OF_PARAMS):
    attributes = attributes + (toolbox.__getattribute__("attribute_" + str(i)),)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator",
                 tools.initCycle,
                 creator.Individual,
                 attributes,
                 n=1)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator",
                 tools.initRepeat,
                 list,
                 toolbox.individualCreator)


# fitness calculation
def mse_fitness_function(individual):
    fitness = hyperparameter_test.get_mse(individual)
    return fitness


toolbox.register("evaluate", mse_fitness_function)

# genetic operators:mutFlipBit

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)

toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0/NUM_OF_PARAMS)


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    print(population)
    start = timeit.default_timer()
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)

    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    # print best solution found:
    print("- Best solution is: \n",
          hyperparameter_test.format_params(hof.items[0]),
          "\n => Mean Squared Error = ",
          hof.items[0].fitness.values[0])

    print('Hall of fame: ', hof)

    # print('Hall of fame2: ', hof.items[0].fitness)
    stop = timeit.default_timer()

    print('Time: ', ((stop - start)/3600), ' hrs')

if __name__ == "__main__":
    main()