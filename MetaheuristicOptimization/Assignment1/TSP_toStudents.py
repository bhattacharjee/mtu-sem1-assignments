#!/usr/bin/env python3

"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random, collections
from Individual import *
import sys
import matplotlib.pyplot as plt
import numpy

myStudentNum = 12345 # Replace 12345 with your student number
random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName:str, _popSize:int, _mutationRate:float, _maxIterations:int):
        """
        Parameters and general variables
        """

        # History of global best and how it changes per run
        self.stat_global_best_history        = []
        # History of mean fitness and how it changes with every run
        self.stat_mean_fitness_history       = []
        # History of best fitness in every run and how it changes
        self.stat_run_best_fitness_history   = []
        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}

        self.readInstance()
        self.initPopulation()

    def updateStats(self):
        thisrun_fitness = [cand.getFitness() for cand in self.population]
        thisrun_best_fitness = max(thisrun_fitness)
        self.stat_run_best_fitness_history.append(thisrun_best_fitness)
        thisrun_mean_fitness = sum(thisrun_fitness) / len(thisrun_fitness)
        self.stat_mean_fitness_history.append(thisrun_mean_fitness)
        self.stat_global_best_history.append(self.best.getFitness())

    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (cid, x, y) = line.split()
            self.data[int(cid)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data,[])
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        self.updateStats()
        print ("Best initial sol: ",self.best.getFitness())

    def updateBest(self, candidate:Individual):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def binaryTournamentSelection(self):
        """
        Your stochastic universal sampling Selection Implementation
        """
        pass

    def uniformCrossover(self, indA:Individual, indB:Individual):
        """
        Your Uniform Crossover Implementation
        """
        pass

    def order1Crossover(self, indA:Individual, indB:Individual):
        """
        Your Order-1 Crossover Implementation
        """
        pass

    def scrambleMutation(self, ind:Individual):
        """
        Your Scramble Mutation implementation
        """
        pass

    def inversionMutation(self, ind:Individual):
        """
        Your Inversion Mutation implementation
        """
        pass

    def crossover(self, indA:Individual, indB:Individual):
        """
        Executes a dummy crossover and returns the genes for a new individual
        """
        midP=int(self.genSize/2)
        cgenes = indA.genes[0:midP]
        for i in range(0, self.genSize):
            if indB.genes[i] not in cgenes:
                cgenes.append(indB.genes[i])
        child = Individual(self.genSize, self.data, cgenes)
        return child

    def reciprocal_index_mutation(self, ind:Individual):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)
        ind.genes[indexA], ind.genes[indexB] = ind.genes[indexB], ind.genes[indexA]

    def mutation(self, ind:Individual):
        if random.random() > self.mutationRate:
            self.reciprocal_index_mutation(ind)
        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        population_fitness = [cand.getFitness() for cand in self.population]
        """
        The smaller the distance of an individual, the more weight it should recieve
        Hence the inversion of the fitness is what we'll use to calculate the probability
        """
        inv_fitness = [1 / (x+1) for x in population_fitness]
        sum_inv_fitness = sum(inv_fitness)
        probabilities = [x / sum_inv_fitness for x in inv_fitness]
        new_pool = numpy.random.choice(self.population, size=len(self.population), p=probabilities, replace=True)

        self.matingPool = []
        for ind_i in new_pool:
            self.matingPool.append( ind_i.copy() )
        duplicates = [item for item, count in collections.Counter(self.matingPool).items() if count > 1]
        print(duplicates)


    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        children = []
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            parent1, parent2 = self.randomSelection()
            child = self.crossover(parent1,parent2)
            self.mutation(child)
            children.append(child)
        assert(len(children) == len(self.population))
        self.population = children

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()
        self.updateStats()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        print ("Total iterations: ", self.iteration)
        print ("Best Solution: ", self.best.getFitness())

def plot_ga(fig, ax, ga, label="None"):
    ax[0].plot(ga.stat_global_best_history, label=label)
    ax[1].plot(ga.stat_run_best_fitness_history, label=label)
    ax[2].plot(ga.stat_mean_fitness_history, label=label)

def main():
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)


    problem_file = sys.argv[1]

    ga = BasicTSP(sys.argv[1], 300, 0.1, 100)
    ga.search()

    fig, ax = plt.subplots(1, 3)
    ax[0].set(title="Global Best", ylabel="Fitness", xlabel="Run")
    ax[1].set(title="Best in this run", ylabel="Fitness", xlabel="Run")
    ax[2].set(title="Average fitness in this run", ylabel="Fitness", xlabel="Run")
    plot_ga(fig, ax, ga, "Basic TSP")
    fig.legend()
    plt.show()

if "__main__" == __name__:
    main()
