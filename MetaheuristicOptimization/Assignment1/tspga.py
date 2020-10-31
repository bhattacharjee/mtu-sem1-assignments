#!/usr/bin/env python3

from lab_tsp_insertion import *
import random, collections, math
from numpy.random import choice

class Instance(object):
    def __init__(self, cities:dict, solution:list=None, identifier:int=0):
        self.cities = self.solution = self.solution_cost = None
        self.id = self.solution_cost = self.fitness_value = None
        if (None == solution):
            self.cities = cities
            self.solution, self.fitness_value = insertion_heuristic1(self.cities)
            self.id = identifier
        else:
            self.cities = cities
            self.solution = solution
            self.fitness_value = -1
        assert(self.validate())

    def validate(self) -> bool:
        for i in self.cities.keys():
            if i not in self.solution:
                return False
        duplicates = [item for item, count in collections.Counter(self.solution).items() if count > 1]
        return [] == duplicates

    def __repr__(self):
        return f"{self.id} : {self.fitness_value} <== {self.solution}"

    def get_solution(self):
        return self.solution

    def get_solution_copy(self) -> int:
        return copy.deepcopy(self.solution)

    def fitness(self) -> int:
        if -1 != self.fitness_value:
            return self.fitness_value
        self.fitness_value = 0
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
        assert(self.fitness_value > 0)
        return self.fitness_value


class GA(object):
    def __init__(self, cities:dict, population_size:int):
        self.cities = cities
        self.population_size = population_size
        self.population = [Instance(cities=cities,identifier=i) for i in range(self.population_size)]
        self.fitness_array = []

    def print_population(self):
        [print(inst) for inst in self.population]

    def calculate_fitness(self):
        self.fitness_array = [inst.fitness() for inst in self.population]
        # Each instance just returns the total distance required, this must be
        # inverted to get the fitness to be used, higher value means more fit
        self.fitness_array = [1/(i+1) for i in self.fitness_array]

    def select_new_parents(self) -> list:
        assert(len(self.population) >= self.population_size)
        if len(self.population) == self.population_size:
            return self.population
        tot_fitness = sum(self.fitness_array)
        probabilities = [i/tot_fitness for i in self.fitness_array]
        choices = choice(self.population, self.population_size, probabilities)
        return choices

    def mate_and_mutate(self, new_parents:list):
        parent_is_mated = [False for i in len(new_parents)]
        children_created = 0
        while True in parent_is_mated or self.population_size != children_created:
            pass
        pass

    def step(self):
        self.calculate_fitness()
        new_parents = self.select_new_parents()
        self.mate_and_mutate(new_parents)


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
    #for inst in ga.population:
    #    print(inst.fitness(), inst.fitness_value)
    ga.step()

if "__main__" == __name__:
    main()
