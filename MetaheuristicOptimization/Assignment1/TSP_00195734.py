#!/usr/bin/python3 -OO

"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
myStudentNum = 195734 # Replace 12345 with your student number
random.seed(myStudentNum)
import numpy
# Since we use numpy's random choice, its seed must be set too
numpy.random.seed(myStudentNum)

import collections, time
from Individual import *
import sys
import matplotlib.pyplot as plt
from lab_tsp_insertion import insertion_heuristic1, insertion_heuristic2
import argparse
#import pprofile


class BasicTSP:
    def __init__(self,
            _fName:str,
            _popSize:int,
            _mutationRate:float,
            _maxIterations:int,\
            mutationType:str="inversion",
            selectionType:str="binarytournament",
            crossoverType:str="order1",
            initPopulationAlgo:str="random"):
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
        self.mutationType   = mutationType.lower()
        self.selectionType  = selectionType.lower()
        self.crossoverType  = crossoverType.lower()
        self.initPopulationAlgo = initPopulationAlgo.lower()
        self.best_update_history = [] # list of tuples (nrun, best)
        self.run_perf_times = []    # How much time did each run take
        self.init_population_perf_time = None
        self.readInstance()
        self.initPopulation()

    def get_description(self)->str:
        string = ""
        string += f"Population Init    : {self.initPopulationAlgo}\n"
        string += f"Mutation Type     : {self.mutationType}\n"
        string += f"Selection Type    : {self.selectionType}\n"
        string += f"Crossover Type   : {self.crossoverType}\n"
        string += f"Population size   : {self.popSize}\n"
        return string

    def print_stats(self):
        print("********************************** STATS **********************************")
        print(f"Genes                           = {self.genSize}")
        print(f"")
        print(f"Run Specifications")
        print(f"      Mutation Type             = {self.mutationType}")
        print(f"      MutationRate              = {self.mutationRate}")
        print(f"      Selection Type            = {self.selectionType}")
        print(f"      Crossover Type            = {self.crossoverType}")
        print(f"      Init Populatin Algo       = {self.initPopulationAlgo}")
        print(f"      Population Size           = {self.popSize}")
        print(f"-----")
        print(f"Performance")
        print(f"Iterations                      = {self.iteration}")
        print(f"Best Fitness                    = {self.best.getFitness()}")
        try:
            print(f"Best Fitness Last Update Run    = {self.best_update_history[-1:][0][0]}")
        except:
            if [] == self.best_update_history:
                print(f"Best Fitness Last Update Run    = 0")
            else:
                raise AssertionError
        print(f"Mean Time Per Step              = {sum(self.run_perf_times) / len(self.run_perf_times)}")
        print(f"Total time for all steps        = {sum(self.run_perf_times)}")
        print(f"Time to initialize population   = {self.init_population_perf_time}")
        print("***************************************************************************")

    def get_stats_dict(self):
        stats = {}
        stats["mean_time_per_iteration"] = sum(self.run_perf_times) / len(self.run_perf_times)
        stats["total_time_for_all_iterations"] = sum(self.run_perf_times)
        stats["time_to_initialize_population"] = self.init_population_perf_time
        stats["total_time_to_run"] = sum(self.run_perf_times) + self.init_population_perf_time
        stats["best_fitness"] = self.best.getFitness()
        try:
            stats["iterations_till_best_fitness"] = self.best_update_history[-1:][0][0]
        except:
            stats["iterations_till_best_fitness"] = 0
        return stats

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

    def initPopulation_random(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data,[])
            individual.computeFitness()
            self.population.append(individual)

    def initPopulation_heuristic1(self):
        for i in range(0, self.popSize):
            solution, cost = insertion_heuristic1(self.data)
            individual = Individual(self.genSize, self.data, solution)
            assert(individual.validate())
            individual.computeFitness()
            self.population.append(individual)

    def initPopulation_heuristic2(self):
        for i in range(0, self.popSize):
            solution, cost = insertion_heuristic2(self.data)
            individual = Individual(self.genSize, self.data, solution)
            assert(individual.validate())
            individual.computeFitness()
            self.population.append(individual)

    def initPopulation(self):
        time1 = time.perf_counter()
        if self.initPopulationAlgo == "random":
            self.initPopulation_random()
        elif self.initPopulationAlgo == "insertionheuristic1":
            self.initPopulation_heuristic1()
        elif self.initPopulationAlgo == "insertionheuristic2":
            self.initPopulation_heuristic2()
        else:
            assert(False)
        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        self.updateStats()
        time1 = time.perf_counter() - time1
        self.init_population_perf_time = time1
        print ("Best initial sol: ",self.best.getFitness())


    def updateBest(self, candidate:Individual):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            self.best_update_history.append((self.iteration, self.best.getFitness()))
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
        x = random.choice(self.matingPool)
        y = random.choice(self.matingPool)
        indA = x if x.getFitness() < y.getFitness() else y
        x = random.choice(self.matingPool)
        y = random.choice(self.matingPool)
        indB = x if x.getFitness() < y.getFitness() else y
        return [indA, indB]

    def selection(self):
        if self.selectionType == "random":
            return self.randomSelection()
        elif self.selectionType == "binarytournament":
            return self.binaryTournamentSelection()
        else:
            assert(False)

    """
    Two alternate implementations
    1.
    fixed_gene_indices = [random.randint(0, self.genSize-1) for i in n_fixed_bits]

    2.
    #while len(fixed_gene_indices) < n_fixed_bits:
        index = random.randint(0, self.genSize-1)
        while index in fixed_gene_indices:
            index = random.randint(0, self.genSize-1)
        fixed_gene_indices.append(index)

    But the implementation using random.sample is more elegant
    """
    def uniformCrossover(self, indA:Individual, indB:Individual):
        """
        Your Uniform Crossover Implementation
        """
        n_fixed_bits = random.randint(5, self.genSize // 2) if 5 < (self.genSize // 2) else 5
        n_fixed_bits = n_fixed_bits if n_fixed_bits <= len(indA.genes) else len(indA.genes)
        # Use set() because lookup is O(1) and we're going to lookup quite a bit
        fixed_gene_indices = frozenset(random.sample(list(range(self.genSize)), n_fixed_bits))
        fixed_genes = frozenset([indA.genes[i] for i in fixed_gene_indices])
        genes_from_par2 = [g for g in indB.genes if g not in fixed_genes]
        child_genes = [g for g in indA.genes]
        # Now overwrite the positions from the other index
        curindex = 0
        for g in genes_from_par2:
            while(curindex in fixed_gene_indices):
                curindex += 1
            child_genes[curindex] = g
            curindex += 1
        child = Individual(self.genSize, self.data, child_genes)
        #print("\n", "*" * 80, "\n", "\nBEFORE CROSSOVER ", indA.genes, "\nAFTER  CROSSOVER ", child.genes, "\nOTHER PARENT     ", indB.genes, "\nFIRST PARENT     ", indA.genes, "\nUNCHANGED ", fixed_genes)
        assert(child.validate())
        return child

    """
    # somewhat inelegant version, a better version below is used
    # Keeping this in case there are bugs, we can revert quickly
    #def order1Crossover2Helper(self, par1:list, par2:list, x:int, y:int)->list:
    #    assert(x <= y)
    #    unchanged = par1[x:(y+1)]
    #    child = []
    #    for i in par2:
    #        if i not in unchanged:
    #            child.append(i)
    #    [child.append(i) for i in unchanged]
    #    assert(len(child) == len(par1))
    #    return child
    """
    def order1Crossover2Helper(self, par1:list, par2:list, x:int, y:int)->list:
        assert(x <= y)
        unchanged = par1[x:(y+1)]
        child = [g for g in par2 if g not in unchanged]
        [child.append(i) for i in unchanged]
        assert(len(child) == len(par1))
        return child

    def order1Crossover2(self, indA:Individual, indB:Individual):
        x = random.randint(0, self.genSize-1)
        y = random.randint(0, self.genSize-1)
        x, y = min(x,y), max(x,y)
        c1 = self.order1Crossover2Helper(indA.genes, indB.genes, x, y)
        c2 = self.order1Crossover2Helper(indB.genes, indA.genes, x, y)
        c1 = Individual(self.genSize, self.data, c1)
        c2 = Individual(self.genSize, self.data, c2)
        assert(c1.validate())
        assert(c2.validate())
        return c1, c2


    """
    # A somewhat inelegant version, better version below is used
    # Keeping this for reference in case there are bugs we can revert quickly
    #def order1Crossover(self, indA:Individual, indB:Individual):
    #    x = random.randint(0, self.genSize-1)
    #    y = random.randint(0, self.genSize-1)
    #    x, y = min(x,y), max(x,y)
    #    child_genes = []
    #    unchanged = indA.genes[x:(y+1)]
    #    for i in indB.genes:
    #        if i not in unchanged:
    #            child_genes.append(i)
    #    [child_genes.append(i) for i in unchanged]
    #    child = Individual(self.genSize, self.data, child_genes)
    #    assert(child.validate())
    #    return child
    """
    def order1Crossover(self, indA:Individual, indB:Individual):
        """
        Your Order-1 Crossover Implementation
        """
        x = random.randint(0, self.genSize-1)
        y = random.randint(0, self.genSize-1)
        x, y = min(x,y), max(x,y)
        unchanged = indA.genes[x:(y+1)]
        child_genes = [g for g in indB.genes if g not in unchanged]
        [child_genes.append(i) for i in unchanged]
        child = Individual(self.genSize, self.data, child_genes)
        assert(child.validate())
        return child

    def scrambleMutation(self, ind:Individual):
        """
        Your Scramble Mutation implementation
        """
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)
        indexA, indexB = min(indexA, indexB), max(indexA, indexB)
        scramble_data= ind.genes[indexA:(indexB+1)]
        random.shuffle(scramble_data)
        for i in range(indexA, (indexB+1)):
            ind.genes[i] = scramble_data[i - indexA]
        assert(ind.validate())

    def inversionMutation(self, ind:Individual):
        """
        Your Inversion Mutation implementation
        """
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)
        indexA, indexB = min(indexA, indexB), max(indexA, indexB)
        data_to_reverse = ind.genes[indexA:(indexB+1)]
        reversed_data = data_to_reverse[::-1]
        for i in range(indexA, (indexB+1)):
            ind.genes[i] = reversed_data[i - indexA]
        assert(ind.validate())

    def dummy_crossover(self, indA:Individual, indB:Individual):
        """
        Executes a dummy crossover and returns the genes for a new individual
        """
        midP=int(self.genSize/2)
        cgenes = indA.genes[0:midP]
        for i in range(0, self.genSize):
            if indB.genes[i] not in cgenes:
                cgenes.append(indB.genes[i])
        child = Individual(self.genSize, self.data, cgenes)
        assert(child.validate())
        return child

    def crossover(self, indA:Individual, indB:Individual):
        child = None
        if self.crossoverType == "order1":
            child = self.order1Crossover(indA, indB)
        elif self.crossoverType == "uniform":
            child = self.uniformCrossover(indA, indB)
        elif self.crossoverType == "dummy":
            child = self.dummy_crossover(indA, indB)
        else:
            assert(False)
        #child.computeFitness()
        return child

    def reciprocal_index_mutation(self, ind:Individual):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)
        ind.genes[indexA], ind.genes[indexB] = ind.genes[indexB], ind.genes[indexA]
        assert(ind.validate())

    def mutation(self, ind:Individual):
        if random.random() > self.mutationRate:
            if self.mutationType == "reciprocal":
                self.reciprocal_index_mutation(ind)
            elif self.mutationType == "inversion":
                self.inversionMutation(ind)
            elif self.mutationType == "scramble":
                self.scrambleMutation(ind)
            else:
                assert(False)
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
        #duplicates = [item for item, count in collections.Counter(self.matingPool).items() if count > 1]
        #print(f"{len(duplicates)} mating_pool={len(self.matingPool)}")


    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        children = []
        if self.crossoverType != "order1variation2":
            for i in range(0, len(self.population)):
                """
                Depending of your experiment you need to use the most suitable algorithms for:
                1. Select two candidates
                2. Apply Crossover
                3. Apply Mutation
                """
                parent1, parent2 = self.selection()
                child = self.crossover(parent1,parent2)
                self.mutation(child)
                children.append(child)
            assert(len(children) == len(self.population))
        else:
            """
            This variation of order 1 crossover produces two children with the
            same set of indices.
            The regular crossover produces just one child, and for the next
            child, a different set of indices are used.
            This check is a little bit out of place here, but I didn't want to
            change the framework too much.
            """
            while len(children) < len(self.population):
                parent1, parent2 = self.selection()
                child1, child2 = self.order1Crossover2(parent1,parent2)
                self.mutation(child1)
                self.mutation(child2)
                children.append(child1)
                children.append(child2)
        self.population = children

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """
        time1 = time.perf_counter()
        self.updateMatingPool()
        self.newGeneration()
        self.updateStats()
        time1 = time.perf_counter() - time1
        self.run_perf_times.append(time1)

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

"""
def plot_ga(fig, ax, ga, label="None"):
    ax[0].plot(ga.stat_global_best_history, label=label)
    ax[1].scatter(list(range(len(ga.stat_run_best_fitness_history))), ga.stat_run_best_fitness_history)
    #ax[2].scatter(list(range(len(ga.stat_mean_fitness_history))), ga.stat_mean_fitness_history)
    ax[2].plot(ga.stat_mean_fitness_history, label=label)
"""
def plot_ga(fig, ax, ga, label="None"):
    ax[0].plot(ga.stat_global_best_history, label=label)
    ax[1].plot(ga.stat_run_best_fitness_history, label=label)
    ax[2].plot(ga.stat_mean_fitness_history, label=label)
    #ax[3].plot(ga.run_perf_times)

def create_and_run_ga(\
        title:str,
        filename:str,
        popsize:int,
        mutationRate:float,
        mutationType:str,
        selectionType:str,
        crossoverType:str,
        initPopulationAlgo:str,
        no_graph,
        runs:int, fig, ax):
    time1 = time.perf_counter()
    ga = BasicTSP(\
            filename,
            popsize,
            mutationRate,
            runs,
            mutationType,
            selectionType,
            crossoverType,
            initPopulationAlgo)
    """
    prof = pprofile.StatisticalProfile()
    with prof(
        period=0.00001, # Sample every 1ms
        single=True, # Only sample current thread
    ):
    """
    ga.search()
    #prof.print_stats()
    time1 = time.perf_counter() - time1
    if not no_graph:
        plot_ga(fig, ax, ga, title)
        fig.suptitle(ga.get_description(), horizontalalignment="left")
    return ga, time1

g_initial_algo = {
        1: "random", 2: "random", 3: "random",
        4: "random", 5: "insertionheuristic1", 6:"insertionheuristic1"}
g_crossover_type = {
        1:"order1", 2:"uniform", 3:"order1",
        4:"uniform", 5:"order1", 6:"uniform"}
g_mutation_type = {
        1:"inversion", 2:"scramble", 3:"scramble",
        4:"inversion", 5:"scramble", 6:"inversion"}


def execute(\
        file_name,
        nruns:int=1,
        pop_size:int=300,
        mutation_rate:float=0.05,
        configuration=1,
        no_graphs=False,
        n_iterations=150):
    global g_initial_algo
    global g_crossover_type
    global g_mutation_type
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    if not no_graphs:
        fig, ax = plt.subplots(1, 3)
        ax[0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[2].set(title="Average fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    # Override population size and mutation rate for BASIC GA (configuration 1
    # and 2)
    if 1 == configuration or 2 == configuration:
        print("BASIC GA: Overriding pop_size = 100 and mutation_rate = 0.05")
        pop_size = 100
        mutation_rate = 0.05
        print("")

    for i in range(nruns):
        ga, t = create_and_run_ga(\
                title="Basic GA - Run %d" % (i,),
                filename=file_name,
                popsize=pop_size,
                mutationRate=mutation_rate,
                mutationType=g_mutation_type[configuration],
                selectionType="binaryTournament",
                crossoverType=g_crossover_type[configuration],
                initPopulationAlgo=g_initial_algo[configuration],
                no_graph=no_graphs,
                runs=n_iterations, fig=fig, ax=ax)
        ga.print_stats()
    print(f"Time taken to run {t}")

    if not no_graphs:
        fig.legend()

def execute_vary_mutation_rate(\
        file_name,
        nruns:int=1,
        pop_size:int=300,
        mutation_rate:float=0.05,
        configuration=1,
        no_graphs=False,
        n_iterations=150,
        mutation_rates=[]):
    global g_initial_algo
    global g_crossover_type
    global g_mutation_type
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    if not no_graphs:
        fig, ax = plt.subplots(1, 3)
        ax[0].set(title="Global Best", ylabel="Fitness", xlabel="Run")
        ax[1].set(title="Best in this run", ylabel="Fitness", xlabel="Run")
        ax[2].set(title="Average fitness in this run", ylabel="Fitness", xlabel="Run")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    # Override population size and mutation rate for BASIC GA (configuration 1
    # and 2)
    if 1 == configuration or 2 == configuration:
        print("BASIC GA: Overriding pop_size = 100 and mutation_rate = 0.05")
        pop_size = 100
        mutation_rate = 0.05
        print("")

    for i in mutation_rates:
        for j in range(nruns):
            ga, t = create_and_run_ga(\
                    title="Run %d - mutation_rate %f" % (j, i,),
                    filename=file_name,
                    popsize=pop_size,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            stats = ga.get_stats_dict()
            print(stats)
            ga.print_stats()
    print(f"Time taken to run {t}")

    if not no_graphs:
        fig.legend()

def execute_vary_population_size(\
        file_name,
        nruns:int=1,
        pop_size:int=300,
        mutation_rate:float=0.05,
        configuration=1,
        no_graphs=False,
        n_iterations=150,
        population_sizes=[]):
    global g_initial_algo
    global g_crossover_type
    global g_mutation_type
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    if not no_graphs:
        fig, ax = plt.subplots(1, 3)
        ax[0].set(title="Global Best", ylabel="Fitness", xlabel="Run")
        ax[1].set(title="Best in this run", ylabel="Fitness", xlabel="Run")
        ax[2].set(title="Average fitness in this run", ylabel="Fitness", xlabel="Run")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    # Override population size and mutation rate for BASIC GA (configuration 1
    # and 2)
    if 1 == configuration or 2 == configuration:
        print("BASIC GA: Overriding pop_size = 100 and mutation_rate = 0.05")
        pop_size = 100
        mutation_rate = 0.05
        print("")

    for i in population_sizes:
        for j in range(nruns):
            ga, t = create_and_run_ga(\
                    title="Run %d - population_size %d" % (j, i,),
                    filename=file_name,
                    popsize=i,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            ga.print_stats()
    print(f"Time taken to run {t}")

    if not no_graphs:
        fig.legend()


def execute_vary_configs(\
        file_name,
        nruns:int=1,
        pop_size:int=300,
        mutation_rate:float=0.05,
        configuration=1,
        no_graphs=False,
        n_iterations=150,
        configs_list=[]):
    global g_initial_algo
    global g_crossover_type
    global g_mutation_type
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    if not no_graphs:
        fig, ax = plt.subplots(1, 3)
        ax[0].set(title="Global Best", ylabel="Fitness", xlabel="Run")
        ax[1].set(title="Best in this run", ylabel="Fitness", xlabel="Run")
        ax[2].set(title="Average fitness in this run", ylabel="Fitness", xlabel="Run")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    for configuration in configs_list:
        for j in range(nruns):
            ga, t = create_and_run_ga(\
                    title="Run %d - configuration %d" % (j, configuration,),
                    filename=file_name,
                    popsize=pop_size,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            ga.print_stats()
    print(f"Time taken to run {t}")

    if not no_graphs:
        fig.legend()


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="File name to parse, str", required=True, type=str)
    parser.add_argument("-ps", "--population-size", help="Population Size", default=300, type=int)
    parser.add_argument("-mr", "--mutation-rate", help="Mutation rate", default=0.05, type=float)
    parser.add_argument("-nr", "--n-runs", help="Number of runs", default=1, type=int)
    parser.add_argument("-ng", "--no-graphs", help="Do not show any graphs", action="store_true")
    parser.add_argument("-c", "--configuration", help="Configuration", choices=[1, 2, 3, 4, 5, 6], default=1, type=int)
    parser.add_argument("-i", "--iterations", help="Number of iterations to perform", default=500, type=int)
    parser.add_argument("-vmr", "--vary-mutation-rate", help="Plot with varying mutation rate, specified as a list", nargs="*", default=[], type=float)
    parser.add_argument("-vps", "--vary-population-size", help="Plot with varying population size, specified as a list", nargs="*", default=[], type=int)
    parser.add_argument("-vc", "--vary-configs", help="Compare different configurations", nargs="*", default=[], type=int)
    args = parser.parse_args()
    filename        = args.file_name
    mutationRate    = args.mutation_rate
    populationSize  = args.population_size
    noGraphs        = args.no_graphs
    n_runs          = args.n_runs
    config          = args.configuration
    niterations     = args.iterations
    print(args.vary_mutation_rate)

    if (None != args.vary_configs and 0 != len(args.vary_configs)):
        if min(args.vary_configs) < 1 or max(args.vary_configs) > 6:
            print("Varying configs, all configs should be between 1 and 6")
            sys.exit(0)

    if 0 != len(args.vary_mutation_rate):
        execute_vary_mutation_rate(file_name=filename,
                nruns=n_runs,
                pop_size=populationSize,
                mutation_rate=mutationRate,
                configuration=config,
                no_graphs=noGraphs,
                n_iterations=niterations,
                mutation_rates=args.vary_mutation_rate)
    elif 0 != len(args.vary_population_size):
        execute_vary_population_size(file_name=filename,
                nruns=n_runs,
                pop_size=populationSize,
                mutation_rate=mutationRate,
                configuration=config,
                no_graphs=noGraphs,
                n_iterations=niterations,
                population_sizes=args.vary_population_size)
    elif None != args.vary_configs and 0 != len(args.vary_configs):
        execute_vary_configs(file_name=filename,
                nruns=n_runs,
                pop_size=populationSize,
                mutation_rate=mutationRate,
                configuration=config,
                no_graphs=noGraphs,
                n_iterations=niterations,
                configs_list=args.vary_configs)
    else:
        execute(file_name=filename,
                nruns=n_runs,
                pop_size=populationSize,
                mutation_rate=mutationRate,
                configuration=config,
                no_graphs=noGraphs,
                n_iterations=niterations)
    try:
        if not noGraphs:
            plt.show()
    except:
        print("Could not show performance graphs")
