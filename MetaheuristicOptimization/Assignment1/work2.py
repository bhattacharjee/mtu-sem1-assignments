#!/usr/bin/python3 -OO

"""
* Important to set -OO. There are a lot of expensive asserts in the code that
* increase runtime anywhere between 2X and 10X.
* Setting -OO means that those asserts won't be called, as this is the release
* configuration
"""

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
from matplotlib.ticker import MaxNLocator
from lab_tsp_insertion import insertion_heuristic1, insertion_heuristic2
import argparse

import statistics
import json
import numpy as np
import pickle
from multiprocessing.pool import ThreadPool, Pool
from heapq import *
#import pprofile

g_run_name = "DEFAULT_RUN"
g_n_processes = 7
g_elitist = False
g_subtract_from_max = True
g_uniform_crossover_large = False
g_order1_crossover_large = False
g_elitist_parents_ratio = 0.1

# If this is set to True, then the mating pool selection will be probabilistic
# based on the fitness of each individual in the pool.
g_weighted_mating_pool = False
g_alternate_mating_pool = True

g_run_num = 0
def get_run_number():
    global g_run_num
    g_run_num += 1
    return g_run_num

def str_to_num(s:str)->int:
    for i, c in enumerate(s):
        if c in list("01234567890"):
            try:
                return int(s[i:])
            except:
                return float(s[i:])

# Class to compare run statistics and print comarative graphs
class CompareRunStats(object):
    def __init__(self):
        self.readings = {}

    def add_results(self, label:str, run_no:int, stats:dict):
        if label not in self.readings.keys():
            self.readings[label] = []
        self.readings[label].append(stats)

    def plot_line_graph(self, field:str, title:str, xlabel:str, ylabel:str, y_lim_zero:bool, pmarker, ax):
        for key in self.readings.keys():
            yaxis = []
            for stat in self.readings[key]:
                yaxis.append(stat[field])
            if None != pmarker:
                ax.plot(yaxis, label=key, marker=pmarker)
            else:
                ax.plot(yaxis, label=key)
        ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
        if y_lim_zero:
            ax.set_ylim(ymin=0)
        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_line_graph2(self, field:str, title:str, xlabel:str, ylabel:str, y_lim_zero:bool, pmarker, ax, lambdafn):
        xxaxis = []
        yyaxis = []
        for key in self.readings.keys():
            yaxis = []
            xxaxis.append(str_to_num(key))
            for stat in self.readings[key]:
                yaxis.append(stat[field])
            yyaxis.append(lambdafn(yaxis))
        d = {}
        for x, y in zip(xxaxis, yyaxis):
            d[x] = y
        xxaxis = sorted(xxaxis)
        yyaxis = [d[x] for x in xxaxis]
        if None != pmarker:
            ax.plot(xxaxis, yyaxis, marker=pmarker)
        else:
            ax.plot(xxaxis, yyaxis)
        ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
        ax.set_ylim(ymin=0)
        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    def bar_chart(self, field:str, title:str, xlabel:str, ylabel:str, laggr, barxlabel, xlabellambda, ax, norotate):
        aggregates = []
        allkeys = []
        for key in self.readings.keys():
            yaxis = []
            for stat in self.readings[key]:
                yaxis.append(stat[field])
            aggregate = laggr(yaxis)
            aggregates.append(aggregate)
            if None == xlabellambda:
                allkeys.append(key)
            else:
                allkeys.append(xlabellambda(key))
        rects = ax.bar(allkeys, aggregates, color="lightgreen")
        if not norotate:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        ax.set(title=title, ylabel=ylabel, xlabel=barxlabel)
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(int(aggregates[i]) if aggregates[i] > 1 else "%0.3f" % float(aggregates[i]),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -30),
                    textcoords="offset points",
                    ha='center', va='bottom', color='black')


    def grouped_bar_chart(self, field:str, title:str, xlabel:str, ylabel:str, laggrfn, laggrfn2, barxlabel, xlabellambda, ax, norotate):
        aggregates = []
        aggregates2 = []
        allkeys = []
        n_x = len(self.readings.keys())
        X = np.arange(n_x)
        width = 0.3
        if None == laggrfn or None == laggrfn2:
            raise AssertionError
        for key in self.readings.keys():
            yaxis = []
            for stat in self.readings[key]:
                yaxis.append(stat[field])
            aggregate = laggrfn(yaxis)
            aggregates.append(aggregate)
            aggregate = laggrfn2(yaxis)
            aggregates2.append(aggregate)
            if None == xlabellambda:
                allkeys.append(key)
            else:
                allkeys.append(xlabellambda(key))
        ax.plot(aggregates, marker='o')
        markers = [f"x={x}\n{int(y)}" for x, y in zip(allkeys, aggregates)]
        rects1 = ax.bar(X, height=aggregates, width=width, color='lightgreen')
        rects2 = ax.bar(X+width, height=aggregates2, width=width, color='orange')
        for i, rect in enumerate(rects1):
            height = rect.get_height()
            #ax.annotate(allkeys[i],
            ax.annotate(markers[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -30),
                    textcoords="offset points",
                    ha='center', va='bottom')
        ax.set(title=title, ylabel=ylabel, xlabel=barxlabel)
        ax.set_xticks([])

    # Process and print graph
    def process(self, description, barxlabel, xbarlabellambda, norotate=False, alternate=False):
        print(f"in process {norotate} {alternate}")
        global g_run_name
        fig, ax = plt.subplots(3, 3)
        fig.suptitle(description)
        """
        if __debug__:
            for key, val in self.readings.items():
                print(key)
                print(json.dumps(val, indent=4))
        """
        if not alternate:
            self.bar_chart(\
                    "total_time_to_run",
                    "MEAN RUN TIME",
                    barxlabel,
                    "time (s)",
                    lambda x: statistics.median(x),
                    barxlabel,
                    xbarlabellambda,
                    ax[0][0],
                    norotate)
        else:
            self.plot_line_graph2(\
                    field="total_time_to_run",
                    title="MEAN TOTAL RUN TIME",
                    xlabel="Run",
                    ylabel="time (s)",
                    y_lim_zero=False,
                    pmarker='.',
                    ax=ax[0][0],
                    lambdafn=lambda x:statistics.mean(x))
        self.plot_line_graph(\
                field="total_time_to_run",
                title="TOTAL RUN TIME",
                xlabel="Run",
                ylabel="time (s)",
                y_lim_zero=False,
                pmarker='.',
                ax=ax[0][1])
        self.bar_chart(\
                "best_fitness",
                "BEST FITNESS",
                barxlabel,
                "Disatnce",
                lambda x: min(x),
                barxlabel,
                xbarlabellambda,
                ax[0][2],
                norotate)
        self.plot_line_graph(\
                field="mean_time_per_iteration",
                title="TIME TO RUN PER ITERATION",
                xlabel="Run",
                ylabel="time (s)",
                y_lim_zero=False,
                pmarker='.',
                ax=ax[1][1])
        if not alternate:
            self.bar_chart(
                    "mean_time_per_iteration",
                    "MEAN TIME PER ITERATION",
                    barxlabel,
                    "time (s)",
                    lambda x: sum(x) / len(x), # min/max anything else
                    barxlabel,
                    xbarlabellambda, #How to modify xlabels
                    ax[1][0],
                    norotate)
        else:
            self.plot_line_graph2(\
                    field="mean_time_per_iteration",
                    title="MEAN TIME PER ITERATION",
                    xlabel="Run",
                    ylabel="time (s)",
                    y_lim_zero=False,
                    pmarker='.',
                    ax=ax[1][0],
                    lambdafn=lambda x:statistics.mean(x))
        self.grouped_bar_chart(\
                "best_fitness",
                "MEDIAN AND MEAN BEST FITNESS",
                barxlabel,
                "Disatnce",
                lambda x: statistics.median(x),
                lambda x: statistics.mean(x),
                barxlabel,
                xbarlabellambda,
                ax[2][0],
                norotate)
        self.plot_line_graph(\
                field="best_fitness",
                title="BEST FITNESS",
                xlabel="Run",
                ylabel="Distance",
                y_lim_zero=False,
                pmarker='.',
                ax=ax[2][1])
        if not alternate:
            self.bar_chart(\
                    "iterations_till_best_fitness",
                    "MEDIAN ITERS TO REACH BEST FITNESS\n(convergence speed)",
                    barxlabel,
                    "Runs",
                    lambda x: statistics.median(x),
                    barxlabel,
                    xbarlabellambda,
                    ax[1][2],
                    norotate)
        else:
            self.plot_line_graph2(\
                    field="iterations_till_best_fitness",
                    title="MEDIAN ITERS TO REACH BEST FITNESS\n(convergence speed)",
                    xlabel="Run",
                    ylabel="time (s)",
                    y_lim_zero=False,
                    pmarker='.',
                    ax=ax[1][2],
                    lambdafn=lambda x: statistics.median(x))
        self.plot_line_graph(\
                field="iterations_till_best_fitness",
                title="ITERS TO REACH BEST FITNESS",
                ylabel="Iterations",
                xlabel="Run",
                y_lim_zero=False,
                pmarker='.',
                ax=ax[2][2])
        fig.legend()
        save_filename=f"{g_run_name}-1.pickle"
        with open(save_filename, "wb") as f:
            pickle.dump(ax, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_filename=f"{g_run_name}_fig-1.pickle"
        with open(save_filename, "wb") as f:
            pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        # Median
        self.stat_median_fitness_history     = []
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
        string += f"Population Init : {self.initPopulationAlgo} "
        string += f"Mutation Type : {self.mutationType}\n"
        string += f"Mutation Rate : {self.mutationRate} "
        string += f"Selection Type : {self.selectionType}\n"
        string += f"Crossover Type : {self.crossoverType} "
        string += f"Population size : {self.popSize}\n"
        string += f"Iterations : {self.iteration}\n"
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
        stats["n_genes"] = self.genSize
        return stats

    def updateStats(self):
        thisrun_fitness = [cand.getFitness() for cand in self.population]
        thisrun_best_fitness = max(thisrun_fitness)
        self.stat_run_best_fitness_history.append(thisrun_best_fitness)
        thisrun_mean_fitness = sum(thisrun_fitness) / len(thisrun_fitness)
        thisrun_median_fitness = statistics.median(thisrun_fitness)
        self.stat_mean_fitness_history.append(thisrun_mean_fitness)
        self.stat_global_best_history.append(self.best.getFitness())
        self.stat_median_fitness_history.append(thisrun_median_fitness)

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
        if __debug__:
            print ("Best initial sol: ",self.best.getFitness())


    def updateBest(self, candidate:Individual):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            self.best_update_history.append((self.iteration, self.best.getFitness()))
            if __debug__:
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
        global g_uniform_crossover_large
        if not g_uniform_crossover_large:
            # Select random number of genes to crossover, minimum of 5
            n_fixed_bits = random.randint(5, self.genSize // 2) if 5 < (self.genSize // 2) else 5
        else:
            # Select randomly between 50% and 75% of genes to crossover
            low = int(self.genSize * 0.5)
            high = int(self.genSize * 0.75)
            n_fixed_bits = random.choice(range(low, high+1))
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

    # Alternate version, this produces two children with the same parents
    # and the same indices, instead of one as with the other version
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

    def getOrder1CrossoverIndices(self):
        """
        Get the indices for order 1 crossover
        """
        global g_order1_crossover_large
        if not g_order1_crossover_large:
            x = random.randint(0, self.genSize-1)
            y = random.randint(0, self.genSize-1)
            return x, y
        else:
            # Select between 50% and 75% of genes
            minToCrossover = int(self.genSize * 0.5)
            maxToCrossover = int(self.genSize * 0.75)
            x = random.randint(0, self.genSize - minToCrossover - 1)
            if x + maxToCrossover > self.genSize - 1:
                maxToCrossover = self.genSize - x - 1
            y = random.randint(x + minToCrossover, x + maxToCrossover)
            return x, y


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
        x, y = self.getOrder1CrossoverIndices()
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
        global g_weighted_mating_pool

        if g_weighted_mating_pool:
            """
            Updating the mating pool before creating a new generation
            """
            inv_fitness = [1 / (0.00001 + cand.getFitness()) for cand in self.population]
            sum_inv_fitness = sum(inv_fitness)
            probabilities = [x / sum_inv_fitness for x in inv_fitness]
            new_pool = numpy.random.choice(self.population, size=len(self.population), p=probabilities, replace=True)
            self.matingPool = []
            for ind_i in new_pool:
                self.matingPool.append( ind_i.copy() )
        else:
            self.matingPool = []
            for ind_i in self.population:
                self.matingPool.append( ind_i.copy() )

    def elitist_replace(self, children):
        global g_elitist_parents_ratio
        if g_elitist:
            temp_mating_pool = []
            inv_fitness = [cand.getFitness() for cand in self.population]
            # heapify is O(n)
            heapify(inv_fitness)
            n_parents = int(self.popSize * g_elitist_parents_ratio) # Keep best parents
            fitness_cutoff = 0
            # This Each iteration runs in O(log(n))
            for i in range(n_parents + 1):
                fitness_cutoff = heappop(inv_fitness)
            for i in self.population:
                if n_parents > 0:
                    if i.getFitness() < fitness_cutoff:
                        temp_mating_pool.append(i)
                        n_parents -= 1
                else:
                    break
            n_children = self.popSize - len(temp_mating_pool)
            if g_subtract_from_max:
                maxfit = max([cand.getFitness() for cand in children])
                inv_fitness = [(maxfit + 1 - cand.getFitness()) for cand in children]
            else:
                inv_fitness = ([1/(0.0001 + cand.getFitness()) for cand in children])
            s = sum(inv_fitness)
            probabilities = [i/s for i in inv_fitness]
            new_pool = numpy.random.choice(children, size=n_children, p=probabilities, replace=False)
            [temp_mating_pool.append(i) for i in new_pool]
            self.population = temp_mating_pool
        else:
            raise AssertionError

    def newGeneration(self):
        global g_alternate_mating_pool, g_elitist, g_subtract_from_max
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
        if g_elitist:
            self.elitist_replace(children)
        else:
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

        if __debug__:
            print ("Total iterations: ", self.iteration)
            print ("Best Solution: ", self.best.getFitness())

"""
def plot_ga(fig, ax, ga, label="None"):
    ax[0].plot(ga.stat_global_best_history, label=label)
    ax[1].scatter(list(range(len(ga.stat_run_best_fitness_history))), ga.stat_run_best_fitness_history)
    #ax[2].scatter(list(range(len(ga.stat_mean_fitness_history))), ga.stat_mean_fitness_history)
    ax[2].plot(ga.stat_mean_fitness_history, label=label)
"""
g_plot_ga_counter = 1
def plot_ga(fig, ax, ga, label="None"):
    global g_run_name, g_plot_ga_counter
    g_plot_ga_counter = g_plot_ga_counter if g_plot_ga_counter != -3 else 1
    g_plot_ga_counter -= 1
    ax[0][0].plot(ga.stat_global_best_history, label=label)
    ax[0][1].plot(ga.stat_run_best_fitness_history, label=label)
    ax[1][0].plot(ga.stat_mean_fitness_history, label=label)
    ax[1][1].plot(ga.stat_median_fitness_history, label=label)
    ax[2][1].set(title="Best Solution", ylabel="Fitness", xlabel="Run")
    rects = ax[2][1].bar(x=label,height=min(ga.stat_global_best_history))
    for i, rect in enumerate(rects):
        height = rect.get_height()
        text = str(int(height)) if height > 1 else "%0.3d" % float(height)
        ax[2][1].annotate(text,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -30 + g_plot_ga_counter * 15),
                textcoords="offset points",
                ha='center', va='bottom')
    ax[2][1].tick_params(labelrotation=20)
    ax[2][0].set(title="Worst Solution", ylabel="Fitness", xlabel="Run")
    ax[2][0].tick_params(labelrotation=20)
    rects = ax[2][0].bar(x=label,height=max(ga.stat_global_best_history))
    for i, rect in enumerate(rects):
        height = rect.get_height()
        text = str(int(height)) if height > 1 else "%0.3d" % float(height)
        ax[2][0].annotate(text,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -30 + g_plot_ga_counter * 15),
                textcoords="offset points",
                ha='center', va='bottom')
    #ax[3].plot(ga.run_perf_times)
    filename = f"{g_run_name}-2.pickle"
    with open(filename, "wb") as f:
        pickle.dump(ax, f, protocol=pickle.HIGHEST_PROTOCOL)
    filename = f"{g_run_name}_fig-2.pickle"
    with open(filename, "wb") as f:
        pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)
    fig.legend()

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

def run_ga(ga, title, i, j, runnum):
    """
    # Since we are running in a multi-process environment, the random
    # seed will be reset when the worker process is forked, and the results
    # are not replicated.
    # I followed Diarmuid's suggestion here to set it to a random number
    # in an AP. With this the results can now be replicated.
    """
    random.seed(myStudentNum + 100 * runnum)
    numpy.random.seed(myStudentNum + 100 * runnum)
    ga.search()
    return ga, title, i, j

def create_ga(\
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
    ga = BasicTSP(\
            filename,
            popsize,
            mutationRate,
            runs,
            mutationType,
            selectionType,
            crossoverType,
            initPopulationAlgo)
    return ga

def plot_ga2(fig, ax, ga, title):
    plot_ga(fig, ax, ga, title)
    fig.suptitle(ga.get_description(), horizontalalignment="left")
    save_filename=f"{g_run_name}-4.pickle"
    with open(save_filename, "wb") as f:
        pickle.dump(ax, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_filename=f"{g_run_name}_fig-4.pickle"
    with open(save_filename, "wb") as f:
        pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
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
    if __debug__:
        print(f"Time taken to run {t}")

    if not no_graphs:
        fig.legend()

def execute_multi_threaded(\
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
    global g_n_processes
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    pool = Pool(g_n_processes)
    futures = []
    t = 0

    # Override population size and mutation rate for BASIC GA (configuration 1
    # and 2)
    if 1 == configuration or 2 == configuration:
        print("BASIC GA: Overriding pop_size = 100 and mutation_rate = 0.05")
        pop_size = 100
        mutation_rate = 0.05
        print("")

    for i in range(nruns):
        thetitle="Basic GA - Run %d" % (i,)
        dummy=0
        ga = create_ga(\
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
        async_result = pool.apply_async(run_ga, (ga, thetitle, i, dummy, get_run_number()))
        futures.append(async_result)
    for async_result in futures:
        async_result.wait()
    for async_result in futures:
        ga, title, i, j = async_result.get()
        ga.print_stats()
        plot_ga2(fig, ax, ga, title)
    if __debug__:
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
    gensize = 0

    crs = CompareRunStats()
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
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
            crs.add_results("MutationRate: %f" % i, j, stats)
            gensize = stats["n_genes"]
            ga.print_stats()
    if __debug__:
        print(f"Time taken to run {t}")

    suptitle = f"EFFECTS OF VARYING MUTATION RATE\n{file_name} configuration={configuration} {g_mutation_type[configuration]} " +\
            f" pop={pop_size} iters={n_iterations} {g_crossover_type[configuration]} initialization={g_initial_algo[configuration]} genesize={gensize}"
    fig.suptitle(suptitle)

    crs.process(\
            suptitle,
            "MUTATION RATE",
            lambda x: "%0.3f" % float(x[len("MutationRate: "):]),
            False,
            True)
    if not no_graphs:
        fig.legend()

def execute_vary_mutation_rate_multi_threaded(\
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
    global g_n_processes

    crs = CompareRunStats()
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    # Override population size and mutation rate for BASIC GA (configuration 1
    # and 2)
    if 1 == configuration or 2 == configuration:
        print("BASIC GA: Overriding pop_size = 100 and mutation_rate = 0.05")
        pop_size = 100
        mutation_rate = 0.05
        print("")

    pool = Pool(g_n_processes)
    futures = []
    t = 0

    gensize = 0
    for i in mutation_rates:
        for j in range(nruns):
            thetitle = "Run %d - mutation_rate %f" % (j, i,)
            ga = create_ga(\
                    title=thetitle,
                    filename=file_name,
                    popsize=pop_size,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            async_result = pool.apply_async(run_ga, (ga, thetitle, i, j, get_run_number()))
            futures.append(async_result)

    for async_result in futures:
        async_result.wait()
    for async_result in futures:
        ga, title, i, j = async_result.get()
        stats = ga.get_stats_dict()
        gensize = stats["n_genes"]
        crs.add_results("MutationRate: %f" % i, j, stats)
        ga.print_stats()
        plot_ga2(fig, ax, ga, title)
    if __debug__:
        print(f"Time taken to run {t}")

    suptitle = f"EFFECTS OF VARYING MUTATION RATE\n{file_name} configuration={configuration} {g_mutation_type[configuration]} " +\
            f" pop={pop_size} iters={n_iterations} {g_crossover_type[configuration]} initialization={g_initial_algo[configuration]} genesize={gensize}"
    fig.suptitle(suptitle)
    crs.process(\
            suptitle,
            "MUTATION RATE",
            lambda x: "%0.3f" % float(x[len("MutationRate: "):]),
            False,
            True)
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

    crs = CompareRunStats()

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    # Override population size and mutation rate for BASIC GA (configuration 1
    # and 2)
    if 1 == configuration or 2 == configuration:
        print("BASIC GA: Overriding pop_size = 100 and mutation_rate = 0.05")
        pop_size = 100
        mutation_rate = 0.05
        print("")

    gensize = 0
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
            stats = ga.get_stats_dict()
            gensize = stats["n_genes"]
            crs.add_results("PopulationSize: %d" % i, j, stats)
            ga.print_stats()
    print(f"Time taken to run {t}")

    suptitle = f"EFFECTS OF VARYING POPULATION SIZE\n{file_name} configuration={configuration} {g_mutation_type[configuration]} " +\
            f" mutationrate={mutation_rate} iters={n_iterations} {g_crossover_type[configuration]} initialization={g_initial_algo[configuration]} genesize={gensize}"
    crs.process(\
            suptitle,
            "POPULATION SIZE",
            lambda x: "%d" % int(x[len("PopulationSize: "):]),
            False,
            True)
    if not no_graphs:
        fig.legend()

def execute_vary_population_size_multi_threaded(\
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
    global g_n_processes
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    crs = CompareRunStats()

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    # Override population size and mutation rate for BASIC GA (configuration 1
    # and 2)
    if 1 == configuration or 2 == configuration:
        print("BASIC GA: Overriding pop_size = 100 and mutation_rate = 0.05")
        pop_size = 100
        mutation_rate = 0.05
        print("")

    pool = Pool(g_n_processes)
    futures = []

    t = "N/A"

    gensize = 0
    for i in population_sizes:
        for j in range(nruns):
            thetitle="Run %d - population_size %d" % (j, i,)
            ga = create_ga(\
                    title=thetitle,
                    filename=file_name,
                    popsize=i,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            async_result = pool.apply_async(run_ga, (ga, thetitle, i, j, get_run_number()))
            futures.append(async_result)
    for async_result in futures:
        async_result.wait()
    for async_result in futures:
        ga, title, i, j = async_result.get()
        stats = ga.get_stats_dict()
        gensize = stats["n_genes"]
        crs.add_results("PopulationSize: %d" % i, j, stats)
        ga.print_stats()
        plot_ga2(fig, ax, ga, title)
    if __debug__:
        print(f"Time taken to run {t}")

    suptitle = f"EFFECTS OF VARYING POPULATION SIZE\n{file_name} configuration={configuration} {g_mutation_type[configuration]} " +\
            f" mutationrate={mutation_rate} iters={n_iterations} {g_crossover_type[configuration]} initialization={g_initial_algo[configuration]} genesize={gensize}"
    crs.process(\
            suptitle,
            "POPULATION SIZE",
            lambda x: "%d" % int(x[len("PopulationSize: "):]),
            False,
            True)
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

    crs = CompareRunStats()

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    gensize = 0
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
            stats = ga.get_stats_dict()
            gensize = stats["n_genes"]
            crs.add_results("Configuration: %d" % configuration, j, stats)
    if __debug__:
        print(f"Time taken to run {t}")

    suptitle = f"EFFECTS OF CONFIGURATIONS\n{file_name} population={pop_size} " +\
            f" mutationrate={mutation_rate} iters={n_iterations} genesize={gensize}"
    crs.process(\
            suptitle,
            "CONFIGURATION",
            lambda x: x[len("Configuration: "):],
            norotate=True)
    if not no_graphs:
        fig.legend()

def execute_vary_configs_multi_threaded(\
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
    global g_n_processes
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    crs = CompareRunStats()

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    pool = Pool(g_n_processes)
    futures = []
    t = 0

    gensize = 0
    for configuration in configs_list:
        for j in range(nruns):
            thetitle = "Run %d - configuration %d" % (j, configuration,)
            ga = create_ga(\
                    title=thetitle,
                    filename=file_name,
                    popsize=pop_size,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            async_result = pool.apply_async(run_ga, (ga, thetitle, configuration, j, get_run_number()))
            futures.append(async_result)
    for async_result in futures:
            async_result.wait()
            print("Wait done")
    for async_result in futures:
        ga, title, configuration, j = async_result.get()
        ga.print_stats()
        stats = ga.get_stats_dict()
        gensize = stats["n_genes"]
        crs.add_results("Configuration: %d" % configuration, j, stats)
        plot_ga2(fig, ax, ga, title)
    if __debug__:
        print(f"Time taken to run {t}")

    suptitle = f"EFFECTS OF CONFIGURATIONS\n{file_name} population={pop_size} " +\
            f" mutationrate={mutation_rate} iters={n_iterations} genesize={gensize}"
    crs.process(\
            suptitle,
            "CONFIGURATION",
            lambda x: x[len("Configuration: "):],
            norotate=True)
    if not no_graphs:
        fig.legend()

def execute_vary_parent_ratio (\
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
    global g_n_processes
    global g_elitist_parents_ratio, g_elitist
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    crs = CompareRunStats()

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    pool = Pool(g_n_processes)
    futures = []
    t = 0

    gensize = 0
    g_elitist = True
    for j in range(nruns):
        for epr in (0.05, 0.1, 0.20, 0.50, 0.80, 0.95):
            g_elitist_parents_ratio = epr
            thetitle = "Run %d - configuration %d" % (epr, configuration,)
            ga = create_ga(\
                    title=thetitle,
                    filename=file_name,
                    popsize=pop_size,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            async_result = pool.apply_async(run_ga, (ga, thetitle, epr, j, get_run_number()))
            futures.append(async_result)
    for async_result in futures:
            async_result.wait()
            print("Wait done")
    for async_result in futures:
        ga, title, configuration, j = async_result.get()
        ga.print_stats()
        stats = ga.get_stats_dict()
        gensize = stats["n_genes"]
        crs.add_results("GenerationGap: %d" % configuration, j, stats)
        plot_ga2(fig, ax, ga, title)
    if __debug__:
        print(f"Time taken to run {t}")

    suptitle = f"EFFECTS OF Varying Generation Gap\n{file_name} population={pop_size} " +\
            f" mutationrate={mutation_rate} iters={n_iterations} genesize={gensize}"
    crs.process(\
            suptitle,
            "VARY POPULATION SIZE",
            lambda x: x[len("GenerationGap: "):],
            norotate=True)
    if not no_graphs:
        fig.legend()

def plot_vary_files(meanTotal, medianTotal, meanPerIteration, medianPerIteration, gene_sizes, files_list, suptitle):
    global g_run_name
    fig, ax = plt.subplots(3, 2)

    def plot(ax, yaxis, xaxis, title):
        # n(Genes) is not in order, need to sort the arrays
        d = {}
        for i, j in zip(xaxis, yaxis):
            d[i] = j
        xaxis = sorted(xaxis)
        yaxis = [d[i] for i in xaxis]
        ax.plot(xaxis, yaxis, marker='.')
        ax.set(title=title, ylabel="Time (s)", xlabel="n(Genes)")

    print(meanTotal, gene_sizes)
    plot(ax[0][0], meanTotal, gene_sizes, "Mean Total Time")
    plot(ax[0][1], medianTotal, gene_sizes, "Median Total Time")
    plot(ax[1][0], meanPerIteration, gene_sizes, "Mean Time Per Iteration")
    plot(ax[1][1], medianPerIteration, gene_sizes, "Median Time Per Iteration")

    fig.suptitle(suptitle)

    save_filename=f"{g_run_name}-3.pickle"
    with open(save_filename, "wb") as f:
        pickle.dump(ax, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_filename=f"{g_run_name}_fig-3.pickle"
    with open(save_filename, "wb") as f:
        pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)


def execute_vary_files(\
        file_name,
        nruns:int=1,
        pop_size:int=300,
        mutation_rate:float=0.05,
        configuration=1,
        no_graphs=False,
        n_iterations=150,
        files_list=[]):
    global g_initial_algo
    global g_crossover_type
    global g_mutation_type
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    crs = CompareRunStats()

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    mean_total_time = []
    mean_time_per_iteration = []
    median_total_time = []
    median_time_per_iteration = []
    gene_sizes = []
    for files in files_list:
        total_time_taken = []
        time_per_iteration_taken = []
        geneSize = -1
        for j in range(nruns):
            ga, t = create_and_run_ga(\
                    title="Run %d - configuration %d" % (j, configuration,),
                    filename=files,
                    popsize=pop_size,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            ga.print_stats()
            stats = ga.get_stats_dict()
            total_time_taken.append(stats["total_time_to_run"])
            time_per_iteration_taken.append(stats["mean_time_per_iteration"])
            crs.add_results("geneSize: %d" % stats["n_genes"], j, stats)
            geneSize = stats["n_genes"]
        mean_total_time.append(statistics.mean(total_time_taken))
        median_total_time.append(statistics.median(total_time_taken))
        mean_time_per_iteration.append(statistics.mean(time_per_iteration_taken))
        median_time_per_iteration.append(statistics.median(time_per_iteration_taken))
        gene_sizes.append(geneSize)

    suptitle = f"EFFECTS OF VARYING POPULATION SIZE\nconfiguration={configuration} {g_mutation_type[configuration]} " +\
            f" mutationrate={mutation_rate} iters={n_iterations} population={pop_size} {g_crossover_type[configuration]} initialization={g_initial_algo[configuration]}"
    plot_vary_files(mean_total_time, median_total_time, mean_time_per_iteration, median_time_per_iteration, gene_sizes, files_list, suptitle)

    if __debug__:
        print(f"Time taken to run {t}")

    crs.process(\
            suptitle,
            "n GENES",
            lambda x: x[len("geneSize: "):],
            True,
            True)
    if not no_graphs:
        fig.legend()

def execute_vary_files_multi_threaded(\
        file_name,
        nruns:int=1,
        pop_size:int=300,
        mutation_rate:float=0.05,
        configuration=1,
        no_graphs=False,
        n_iterations=150,
        files_list=[]):
    global g_initial_algo
    global g_crossover_type
    global g_mutation_type
    global g_n_processes
    if len(sys.argv) < 2:
        print ("Error - Incorrect input")
        print ("Expecting python BasicTSP.py [instance] ")
        sys.exit(0)

    crs = CompareRunStats()
    t = 0
    futures = []
    pool = Pool(g_n_processes)

    if not no_graphs:
        fig, ax = plt.subplots(3, 2)
        ax[0][0].set(title="Global Best", ylabel="Fitness", xlabel="Iteration")
        ax[0][1].set(title="Best in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][0].set(title="Mean fitness in this run", ylabel="Fitness", xlabel="Iteration")
        ax[1][1].set(title="Median fitness in this run", ylabel="Fitness", xlabel="Iteration")
        #ax[3].set(title="Time per step", ylabel="Time", xlabel="Run")

    mean_total_time = []
    mean_time_per_iteration = []
    median_total_time = []
    median_time_per_iteration = []
    gene_sizes = []
    for files in files_list:
        total_time_taken = []
        time_per_iteration_taken = []
        geneSize = -1
        for j in range(nruns):
            thetitle = "Run %d - configuration %d" % (j, configuration,)
            ga = create_ga(\
                    title=thetitle,
                    filename=files,
                    popsize=pop_size,
                    mutationRate=mutation_rate,
                    mutationType=g_mutation_type[configuration],
                    selectionType="binaryTournament",
                    crossoverType=g_crossover_type[configuration],
                    initPopulationAlgo=g_initial_algo[configuration],
                    no_graph=no_graphs,
                    runs=n_iterations, fig=fig, ax=ax)
            async_result = pool.apply_async(run_ga, (ga, thetitle, files, j, get_run_number()))
            futures.append(async_result)
    for async_result in futures:
        async_result.wait()
    for async_result in futures:
        ga, title, files, j = async_result.get()
        ga.print_stats()
        stats = ga.get_stats_dict()
        plot_ga2(fig, ax, ga, title)
        total_time_taken.append(stats["total_time_to_run"])
        time_per_iteration_taken.append(stats["mean_time_per_iteration"])
        crs.add_results("geneSize: %d" % stats["n_genes"], j, stats)
        geneSize = stats["n_genes"]
        mean_total_time.append(statistics.mean(total_time_taken))
        median_total_time.append(statistics.median(total_time_taken))
        mean_time_per_iteration.append(statistics.mean(time_per_iteration_taken))
        median_time_per_iteration.append(statistics.median(time_per_iteration_taken))
        gene_sizes.append(geneSize)

    suptitle = f"EFFECTS OF VARYING POPULATION SIZE\nconfiguration={configuration} {g_mutation_type[configuration]} " +\
            f" mutationrate={mutation_rate} iters={n_iterations} population={pop_size} {g_crossover_type[configuration]} initialization={g_initial_algo[configuration]}"
    plot_vary_files(mean_total_time, median_total_time, mean_time_per_iteration, median_time_per_iteration, gene_sizes, files_list, suptitle)

    if __debug__:
        print(f"Time taken to run {t}")

    crs.process(\
            suptitle,
            "n GENES",
            lambda x: x[len("geneSize: "):],
            True,
            True)
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
    parser.add_argument("-vf", "--vary-files", help="Compare across different files (gene size)", nargs="*", default=[], type=str)
    parser.add_argument("-name", "--run-name", help="Name of run, matplotlib pickles will be saved with this name", default="DEFAULT_RUN", type=str)
    parser.add_argument("-mt", "--multi-threaded", help="Run multi-threaded versions", action="store_true", default=False)
    parser.add_argument("-ucl", "--uniform_crossover-large", help="Choose between50 and 75pc of genes for uniform crossover", action="store_true", default=False)
    parser.add_argument("-ocl", "--order-one-crossover-large", help="Choose between50 and 75pc of genes for uniform crossover", action="store_true", default=False)
    parser.add_argument("-epr", "--elitist-parents-ratio", help="ratio of parents to choose for elitism, between 0 and 0.99, negative specifies no elitism", type=float, default=-1.0)
    parser.add_argument("-vepr", "--vary-elitist-parents-ratio", help="Veary elitist parents ratio from 0.05 0.10 0.25 0.30 0.35 0.40", action="store_true", default=False)

    args = parser.parse_args()
    filename        = args.file_name
    mutationRate    = args.mutation_rate
    populationSize  = args.population_size
    noGraphs        = args.no_graphs
    n_runs          = args.n_runs
    config          = args.configuration
    niterations     = args.iterations

    g_run_name = args.run_name

    if (None != args.vary_configs and 0 != len(args.vary_configs)):
        if min(args.vary_configs) < 1 or max(args.vary_configs) > 6:
            print("Varying configs, all configs should be between 1 and 6")
            sys.exit(0)

    g_uniform_crossover_large = True if args.uniform_crossover_large else False
    g_order1_crossover_large = True if args.order_one_crossover_large else False

    if args.elitist_parents_ratio > 0.0:
        g_elitist_parents_ratio = args.elitist_parents_ratio
        g_elitist = True
    else:
        g_elitist = False

    if g_elitist_parents_ratio > 0.99:
        print("\nElitist parents ratio should be betwen 0 and 0.99\n\n")
        parser.print_help()
        sys.exit(1)

    if 0 != len(args.vary_mutation_rate):
        if args.multi_threaded:
            execute_vary_mutation_rate_multi_threaded(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations,
                    mutation_rates=args.vary_mutation_rate)
        else:
            execute_vary_mutation_rate(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations,
                    mutation_rates=args.vary_mutation_rate)
    elif 0 != len(args.vary_population_size):
        if args.multi_threaded:
            execute_vary_population_size_multi_threaded(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations,
                    population_sizes=args.vary_population_size)
        else:
            execute_vary_population_size(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations,
                    population_sizes=args.vary_population_size)
    elif None != args.vary_configs and 0 != len(args.vary_configs):
        if args.multi_threaded:
            execute_vary_configs_multi_threaded(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations,
                    configs_list=args.vary_configs)
        else:
            execute_vary_configs(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations,
                    configs_list=args.vary_configs)
    elif None != args.vary_files and 0 != len(args.vary_files):
        if args.multi_threaded:
            execute_vary_files_multi_threaded(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations,
                    files_list=args.vary_files)
        else:
            execute_vary_files(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations,
                    files_list=args.vary_files)
    elif args.vary_elitist_parents_ratio:
        print("Vary elitist parent ratio")
        g_elitist = True
        execute_vary_parent_ratio(file_name=filename,
                nruns=n_runs,
                pop_size=populationSize,
                mutation_rate=mutationRate,
                configuration=config,
                no_graphs=noGraphs,
                n_iterations=niterations,
                configs_list=args.vary_configs)
    else:
        if args.multi_threaded:
            execute_multi_threaded(file_name=filename,
                    nruns=n_runs,
                    pop_size=populationSize,
                    mutation_rate=mutationRate,
                    configuration=config,
                    no_graphs=noGraphs,
                    n_iterations=niterations)
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
            pass
            plt.show()
    except:
        print("Could not show performance graphs")
