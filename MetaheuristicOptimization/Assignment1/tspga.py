#!/usr/bin/env python3

from lab_tsp_insertion import *
import random, collections, math
from numpy.random import choice

def validate_array(arr):
    duplicates = [item for item, count in collections.Counter(arr).items() if count > 1]
    return [] == duplicates

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
            self.id = identifier
        assert(self.validate())

    def validate(self) -> bool:
        for i in self.cities.keys():
            if i not in self.solution:
                return False
        duplicates = [item for item, count in collections.Counter(self.solution).items() if count > 1]
        return [] == duplicates

    def __repr__(self):
        return f"{self.id} : {self.fitness_value} <== {self.solution}"

    def __eq__(self, other):
        assert(isinstance(other, Instance))
        return self.solution == other.solution

    def get_solution(self):
        return self.solution

    def get_solution_copy(self) -> int:
        return copy.deepcopy(self.solution)

    def get_total_distance(self) -> int:
        self.fitness()
        return self.fitness_value

    def fitness(self) -> int:
        if -1 != self.fitness_value:
            return 1.0 / (1 + self.fitness_value)
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
        return 1.0 / (self.fitness_value + 1)




class GA(object):
    def __init__(self, cities:dict, population_size:int, crossover_fn):
        self.cities = cities
        self.population_size = population_size
        self.instance_count = 0
        self.step_count = 0
        self.crossover_fn = crossover_fn
        self.fitness_array = []
        self.best = None
        self.population = [Instance(cities=cities,identifier=self.get_instance_count())\
                                for i in range(self.population_size)]

    def get_instance_count(self):
        self.instance_count += 1
        return self.instance_count

    def print_population(self):
        [print(inst) for inst in self.population]

    def calculate_fitness(self):
        self.fitness_array = [inst.fitness() for inst in self.population]

    def get_mating_pool(self, size) -> list:
        assert(len(self.population) >= self.population_size)
        tot_fitness = sum(self.fitness_array)
        probabilities = [i/tot_fitness for i in self.fitness_array]
        choices = choice(self.population, size, probabilities)
        return choices

    def binary_tournament_selection(self, mating_pool:list):
        s1 = random.choice(mating_pool)
        s2 = random.choice(mating_pool)
        #print('*' * 100, "\n", s1, "\n", s2)
        while s1 == s2:
            s1 = random.choice(mating_pool)
        f1 = s1.fitness()
        f2 = s2.fitness()
        return s1 if f1 > f2 else s2

    def crossover(self, p1, p2):
        parr1 = p1.solution
        parr2 = p2.solution
        children = self.crossover_fn(parr1, parr2)
        return children

    def mutate(self, child:list, probability=float)->list:
        r = random.uniform(0,1)
        if (r > probability):
            return child
        length = len(child)
        x = y = random.choice(range(length))
        while x == y:
            y = random.choice(range(length))
        (child[x], child[y]) = (child[y], child[x])
        return child

    def mate_and_mutate(self, mating_pool:list):
        parent_is_mated = [False for i in range(len(mating_pool))]
        children = []
        children_created = 0
        while children_created < self.population_size:
            p1 = self.binary_tournament_selection(mating_pool)
            p2 = self.binary_tournament_selection(mating_pool)
            #while p1 == p2:
            #    p2 = self.binary_tournament_selection(mating_pool)
            tchildren = self.crossover(p1, p2)
            children_created += len(tchildren)
            for child in tchildren:
                child = self.mutate(child, 0.5)
                inst = Instance(cities=self.cities, solution=child, identifier=self.get_instance_count())
                children.append(inst)
            assert(len(children) > 0)
        return children

    def print_step_result(self, parents, children):
        max_fitness = -1
        for inst in parents:
            f = inst.fitness()
            if f > max_fitness:
                max_fitness = f
                self.best = inst
        for inst in children:
            f = inst.fitness()
            if f > max_fitness:
                max_fitness = f
                self.best = inst
        #print(f"Results from iteration {self.step_count}: {self.best.fitness()} {self.best}")
        print(f"Results from iteration {self.step_count}: {self.best.fitness()} {self.best.get_total_distance()}")

    def step(self):
        self.step_count += 1
        self.calculate_fitness()
        mating_pool = self.get_mating_pool(self.population_size)
        children = self.mate_and_mutate(mating_pool)
        self.print_step_result(self.population, children)
        self.population = children

# Select the unchanged from par1, and then jumble up par2
def order_one_crossover_helper(par1:list, par2:list, x, y)->list:
    unchanged = par1[x:(y+1)]
    child = []
    for i in par2:
        if i not in unchanged:
            child.append(i)
    [child.append(i) for i in unchanged]
    assert(len(child) == len(par1) and validate_array(child))
    return child

def order_one_crossover(parr1:list, parr2:list)->tuple:
    assert(len(parr1) == len(parr2))
    length = len(parr1)
    x = random.choice(range(length))
    y = random.choice(range(length))
    x, y = min(x, y), max(x, y)
    c1 = order_one_crossover_helper(parr1, parr2, x, y)
    c2 = order_one_crossover_helper(parr2, parr1, x, y)
    return c1, c2

def get_cities(directory):
    allcities = {}
    for filename in os.listdir(directory):
        cities = readInstance(directory + os.path.sep + filename)
        allcities = {**allcities, **cities}
    return allcities

def main():
    directory = "small"# sys.argv[1]
    popsize = 10 if len(sys.argv) <= 2 else int(sys.argv[2])
    allcities = get_cities(directory)
    ga = GA(allcities, popsize, crossover_fn=order_one_crossover)
    #for inst in ga.population:
    #    print(inst.fitness(), inst.fitness_value)
    for i in range(500):
        ga.step()

if "__main__" == __name__:
    main()
