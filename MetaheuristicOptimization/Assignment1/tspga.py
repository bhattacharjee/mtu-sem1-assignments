#!/usr/bin/env python3

from lab_tsp_insertion import *
import random, collections, math

class Instance(object):
    def __init__(self, cities:dict, identifier:int=0):
        self.cities = cities
        self.solution, self.solution_cost = insertion_heuristic1(self.cities)
        self.id = identifier
        self.solution_cost
        self.fitness_value = -1

    """
    def __init__(self, cities:dict, solution:list, identifier:int=0):
        self.cities = cities
        self.solution = solution
        self.solution_cost = -1
        self.fitness_value = -1
    """

    def __repr__(self):
        return f"{self.id} : {self.solution_cost} <== {self.solution}"

    def get_solution(self):
        return self.solution

    def get_solution_copy(self):
        return copy.deepcopy(self.solution)

    def fitness(self):
        if -1 != self.fitness_value:
            return self.fitness_value
        self.fitness_value = 0
        # 1 to N-1
        for i in range(1, len(self.solution)):
            c1, c2 = self.solution[i-1], self.solution[i]
            x1, y1 = self.cities[c1]
            x2, y2 = self.cities[c2]
            dist = round(math.sqrt((x2-x1)**2 + (y2-y1)**2))
            self.fitness_value += dist
        last_city = self.solution[len(self.solution)-1]
        first_city = self.solution[0]
        x1, y1 = self.cities[last_city]
        x2, y2 = self.cities[first_city]
        dist = round(math.sqrt((x2-x1)**2 + (y2-y1)**2))
        self.fitness_value += dist
        return self.fitness_value


class GA(object):
    def __init__(self, cities:dict, population_size:int):
        self.cities = cities
        self.population_size = population_size
        self.population = []
        self.population = [Instance(cities,i) for i in range(self.population_size)]

    def print_population(self):
        [print(inst) for inst in self.population]


def get_cities(directory):
    allcities = {}
    for filename in os.listdir(directory):
        cities = readInstance(directory + os.path.sep + filename)
        allcities = {**allcities, **cities}
    return allcities

def main():
    directory = sys.argv[1]
    popsize = 10 if len(sys.argv) <= 2 else int(sys.argv[2])
    allcities = get_cities(directory)
    ga = GA(allcities, popsize)
    for inst in ga.population:
        print(inst.fitness(), inst.solution_cost)
    """
    for i in range(runs):
        for filename in os.listdir(directory):
            cities = readInstance(directory + os.path.sep + filename)
            print(cities)
            allcities = {**allcities, **cities}
            solutionPath, solutionCost = insertion_heuristic1(cities)
            print(solutionPath, solutionCost)
    """

if "__main__" == __name__:
    main()
