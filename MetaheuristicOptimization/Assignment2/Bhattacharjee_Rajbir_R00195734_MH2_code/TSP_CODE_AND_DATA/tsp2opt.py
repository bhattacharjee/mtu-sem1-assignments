#!/usr/bin/env python3

import sys
import math
import argparse
from functools import lru_cache
import matplotlib.pyplot as plt
from lab_tsp_insertion import *

class TSPSolution(object):
    def __init__(self, instance:list, tour:list, distance:int, use_cache:bool=True):
        self.tour = tour
        self.n_cities = len(tour)
        self.inst = instance
        self.use_cache = use_cache
        self.distance = self.calculate_solution_distance()

    def __repr__(self):
        return f"Tour={self.tour}, distance={self.distance}, n_cities={self.n_cities}"

    def get_n_cities(self):
        return self.n_cities

    @lru_cache(maxsize=1024 * 1024 * 1024)
    def get_distance_lru(self, x, y):
        # This function takes the city number, not index into array
        c1x, c1y = self.inst[x]
        c2x, c2y = self.inst[y]
        return math.sqrt((c2x - c1x) * (c2x - c1x) + (c2y - c1y) * (c2y - c1y))

    def get_distance_nocache(self, x, y):
        # This function takes the city number, not index into array
        c1x, c1y = self.inst[x]
        c2x, c2y = self.inst[y]
        return math.sqrt((c2x - c1x) * (c2x - c1x) + (c2y - c1y) * (c2y - c1y))

    def get_distance(self, x, y):
        if self.use_cache:
            return self.get_distance_lru(x, y)
        else:
            return self.get_distance_nocache(x, y)

    def copy(self):
        return TSPSolution(self.inst, self.tour.copy(), self.distance, self.use_cache)

    @lru_cache(maxsize=1024 * 1024 * 1024)
    def is_valid_swap_lru(self, x:int, y:int):
        x, y = min(x, y), max(x, y)
        x1 = (x + 1) % self.n_cities
        y1 = (y + 1) % self.n_cities
        return x != y and x != y1 and x != x1 and y != y1 and y != x1 and y1 != x1

    def is_valid_swap_nocache(self, x:int, y:int):
        x, y = min(x, y), max(x, y)
        x1 = (x + 1) % self.n_cities
        y1 = (y + 1) % self.n_cities
        return x != y and x != y1 and x != x1 and y != y1 and y != x1 and y1 != x1

    def is_valid_swap(self, x:int, y:int):
        # can two edges be swapped at all?
        if self.use_cache:
            return self.is_valid_swap_lru(x, y)
        else:
            return self.is_valid_swap_nocache(x, y)

    def calculate_cost_if_swapped(self, x:int, y: int):
        # If the edges are swapped what would be the cost
        x, y = min(x, y), max(x, y)
        x1 = (x + 1) % self.n_cities
        y1 = (y + 1) % self.n_cities
        xx1 = self.get_distance(self.tour[x], self.tour[x1])
        xy = self.get_distance(self.tour[x], self.tour[y])
        yy1 = self.get_distance(self.tour[y], self.tour[y1])
        x1y1 = self.get_distance(self.tour[x1], self.tour[y1])
        return self.distance - xx1 - yy1 + x1y1 + xy

    def get_cache_stats(self):
        return self.get_distance.cache_info(), self.is_valid_swap.cache_info()

    def calculate_solution_distance(self):
        dist = 0
        for x in range(self.n_cities):
            y = (x + 1) % self.n_cities
            dist += self.get_distance(self.tour[x], self.tour[y])
        return dist

    def perform_swap(self, x, y):
        # Actually perform the swap, this changes state
        x, y = min(x, y), max(x, y)
        new_distance = self.calculate_cost_if_swapped(x, y)
        x1 = (x + 1) % self.n_cities
        y1 = (y + 1) % self.n_cities
        assert(x1 < y)
        self.tour[x1], self.tour[y] = self.tour[y], self.tour[x1]
        i = x1 + 1
        j = y - 1
        while (i < j):
            self.tour[i], self.tour[j] = self.tour[j], self.tour[i]
            i += 1
            j -= 1
        #if int(self.calculate_solution_distance()) != int(new_distance):
        #    print(f"New distance should have been {new_distance} but is {self.calculate_solution_distance()}")
        self.distance = new_distance

class TSPHillClimbing(object):
    def __init__(self, inst:dict = None, max_sideways_moves:int=0,\
            description:str="None", time_plot=None, distance_plot=None,\
            distance_time_plot=None, verbose:bool=False,\
            use_random_heuristic:bool=False, use_cache:bool=True):
        self.inst = inst
        self.ind = None
        self.g_best_solution = None
        self.g_best_distance = 99999999999999999999999999999999999999999
        self.current_iter_sols = []
        self.current_iter_dist = None
        self.max_sideways_moves = max_sideways_moves
        self.n_sideways_moves = 0
        self.g_best_distance = 999999999999999999999999999999999999999999
        self.g_best_sol = None
        self.last_improving_iteration = -1
        self.iteration = -1
        self.figure = None
        self.g_iteration = 0
        self.g_iteration_list = []
        self.time_plot = time_plot
        self.distance_plot = distance_plot
        self.n_restart = -1
        self.description = description
        self.verbose = verbose
        self.use_cache = use_cache
        self.distance_time_plot = distance_time_plot
        self.use_random_heuristic = use_random_heuristic
        self.run_start_time = 0
        self.rt = []              # Array containing the run times (since this restart)
        self.rt2 = []             # Array containing run times (since beginning of run)
        self.y = []               # Array containing the distance at each iteration
        self.iters_list = []      # Array containing the number of iterations
        self.best_dist_hist = []  # Array containing best distances
        if None != self.inst:
            self.ind = self.get_solution()
            self.update_best_g_instance(self.ind)
        self.run_start_time = time.process_time()


    def __repr__(self):
        return f"inst = {self.inst}\n\nindividual = {self.ind}\n\n" +\
                f"g_best_solution = {self.g_best_solution}\n\ng_best_distance = {self.g_best_distance}"

    def update_best_g_instance(self, instance: TSPSolution):
        if self.g_best_distance > instance.distance:
            self.g_best_solution = instance.copy()
            self.g_best_distance = instance.distance

    def get_solution(self):
        if not self.use_random_heuristic:
            cities, distance = insertion_heuristic1(self.inst)
        else:
            cities, distance = randomTours(self.inst)
        # It is important to return a new instance of the class, instead of just
        # modifying the current instance because we're using the LRU cache. This
        # will cause the cache to be invalidated (as the self argument to each
        # function will now be different).
        return TSPSolution(self.inst, cities, distance, use_cache=self.use_cache)

    def check_improving_move(self):
        # Find if there is an improving move possible, store all such moves
        # in current_iter_sols
        self.current_iter_dist = self.ind.distance
        self.current_iter_sols = []
        for i in range(self.ind.get_n_cities()):
            for j in range(i+2, self.ind.get_n_cities()):
                if not self.ind.is_valid_swap(i, j):
                    pass
                else:
                    newcost = self.ind.calculate_cost_if_swapped(i, j)
                    if newcost == self.current_iter_dist:
                        self.current_iter_sols.append(list(i, j))
                    elif newcost < self.current_iter_dist:
                        self.current_iter_dist = newcost
                        self.current_iter_sols = [[i,j]]

    def iterate_once(self, allow_sideways=False):
        # One iteration, this will be called multiple times
        update = False
        old_distance = self.ind.distance
        self.check_improving_move()
        if old_distance == self.current_iter_dist:
            # There are two variants here, where sidewyas can be allowed
            # or where sideways are not allowed
            if not allow_sideways or len(self.current_iter_sols) == 0:
                #print("No moves possible")
                return old_distance, self.current_iter_dist, True
            else:
                if self.verbose:
                    print("Performing sideways move", self.n_sideways_moves + 1)
                self.n_sideways_moves += 1
                update = True
        if int(self.ind.distance) > int(self.current_iter_dist):
            # An improving move has been found
            self.last_improving_iteration = self.iteration
            self.n_sideways_moves = 0
            update = True
        if update:
            toswap = self.current_iter_sols[random.randint(0, len(self.current_iter_sols)-1)]
            x, y = tuple(toswap)
            self.ind.perform_swap(x, y)
            self.update_best_g_instance(self.ind)
        return old_distance, self.current_iter_dist, False

    def iterate(self, n_iterations, allow_sideways=False, max_sideways_moves=-1):
        self.y = []
        self.rt = []
        self.rt2 = []
        self.best_dist_hist = []
        self.iters_list = []
        self.g_iteration_list = []
        self.n_sideways_moves = 0
        self.last_improving_iteration = 0
        t1 = time.process_time()
        for self.iteration in range(n_iterations):
            self.g_iteration += 1
            if self.verbose:
                sys.stdout.write('-')
            old_distance, current_iter_dist, all_moves_worse = self.iterate_once(allow_sideways)

            # Most of the code below is just to print the data
            if 0 == self.iteration:
                self.y.append(old_distance)
            else:
                self.y.append(current_iter_dist)
            tt = time.process_time()
            self.rt.append(tt - t1)
            self.rt2.append(tt - self.run_start_time)
            self.best_dist_hist.append(self.g_best_distance)
            self.iters_list.append(self.iteration)
            self.g_iteration_list.append(self.g_iteration)

            if old_distance == current_iter_dist:
                if self.verbose:
                    print(self.n_sideways_moves)
                if all_moves_worse or not allow_sideways or max_sideways_moves < self.n_sideways_moves:
                    if allow_sideways:
                        if self.verbose:
                            print("Reached maximum sideways moves")
                    break
            if old_distance - current_iter_dist != 0:
                if self.verbose:
                    sys.stdout.write("%f %f" % \
                            ((old_distance - current_iter_dist), current_iter_dist,))
        if self.verbose:
            print("iterations done: ", self.iteration)
            print('-' * 80)
        description = self.description if None != self.description else ""
        if self.distance_plot:
            self.distance_plot.plot(self.iters_list, self.y, label=('%s %d' % (description, self.n_restart)))
        if self.time_plot:
            self.time_plot.plot(self.iters_list, self.rt, label=('%s %d' % (description, self.n_restart)))
        if self.distance_time_plot:
            self.distance_time_plot.plot(self.rt, self.y, label=('%s %d' % (description, self.n_restart)))
        for i,r,d,r2,bd,i2 in zip(self.iters_list, self.rt, self.y, self.rt2, self.best_dist_hist, self.g_iteration_list):
            print(f"{self.n_restart},{i},{r},{d},{r2},{bd},{i2}")

    def restart_and_iterate(self, n_iterations=100, n_restarts:int=5,\
            allow_sideways=False, max_sideways_moves=-1):
        self.run_start_time = time.process_time()
        self.rt2 = []
        print("Restart,Iteration,RunTime,Distance,RunTimeSinceBeginningOfRun,BestDistance,gIterations")
        for self.n_restart in range(n_restarts):
            self.ind = self.get_solution()
            self.update_best_g_instance(self.ind)
            self.iterate(n_iterations, allow_sideways, max_sideways_moves)
            if self.current_iter_dist < self.g_best_distance:
                self.update_best_g_instance(self.ind)

class TSPHillClimbingRandomIprovement(TSPHillClimbing):
    """
    This class picks an edge at random, and then checks all other edges whether
    swapping with that edge produces better results or not
    The functions check_improving_move and iterate have been modified, the rest
    of the code remains the same as its parent class
    """
    def check_improving_move(self):
        self.current_iter_dist = self.ind.distance
        self.current_iter_sols = []
        # Choose an edge at random
        i = random.randint(0, self.ind.get_n_cities() - 1)
        # Iterate through all other nodes to find if they improve the cost
        for j in range(self.ind.get_n_cities()):
            if not self.ind.is_valid_swap(i, j):
                pass
            else:
                newcost = self.ind.calculate_cost_if_swapped(i, j)
                if newcost == self.current_iter_dist:
                    self.current_iter_sols.append(list(i, j))
                elif newcost < self.current_iter_dist:
                    self.current_iter_dist = newcost
                    self.current_iter_sols = [[i,j]]

    def iterate(self, n_iterations, allow_sideways=False, max_sideways_moves=-1):
        self.y = []
        self.rt = []
        self.rt2 = []
        self.g_iteration_list = []
        self.best_dist_hist = []
        self.iters_list = []
        self.n_sideways_moves = 0
        t1 = time.process_time()
        for self.iteration in range(n_iterations):
            self.g_iteration += 1
            if self.verbose:
                sys.stdout.write('-')
            old_distance, current_iter_dist, all_moves_worse = self.iterate_once(allow_sideways)
            if 0 == self.iteration:
                self.y.append(old_distance)
            else:
                self.y.append(current_iter_dist)
            tt = time.process_time()
            self.rt.append(tt - t1)
            self.rt2.append(tt - self.run_start_time)
            self.best_dist_hist.append(self.g_best_distance)
            self.iters_list.append(self.iteration)
            self.g_iteration_list.append(self.g_iteration)

            if old_distance == current_iter_dist:
                # If 500 non-improving moves have been made, then stop
                if self.iteration - self.last_improving_iteration > 500:
                    if self.verbose:
                        print("Reached limit on number of non improving iterations")
                    break
            if old_distance - current_iter_dist > 0:
                if self.verbose:
                    sys.stdout.write("%f %f" % \
                            ((old_distance - current_iter_dist), current_iter_dist))
        if self.verbose:
            print("iterations done: ", self.iteration)
            print('-' * 80)
        description = self.description if None != self.description else ""
        if self.distance_plot:
            self.distance_plot.plot(self.iters_list, self.y, label=('%s %d' % (description, self.n_restart)))
        if self.time_plot:
            self.time_plot.plot(self.iters_list, self.rt, label=('%s %d' % (description, self.n_restart)))
        if self.distance_time_plot:
            self.distance_time_plot.plot(self.rt, self.y, label=('%s %d' % (description, self.n_restart)))
        print("Description = ", self.description)
        print("Restart,Iteration,RunTime,Distance")
        for i,r,d,r2,bd,i2 in zip(self.iters_list, self.rt, self.y, self.rt2, self.best_dist_hist,self.g_iteration_list):
            print(f"{self.n_restart},{i},{r},{d},{r2},{bd},{i2}")
    pass


class TSPFirstImprovement(TSPHillClimbingRandomIprovement):
    """
    This final version makes the following modification:
    1. Choose an edge at random
    2. Iterate through all other edges to check if the combination can be replaced
    3. Whenever a better choice is found, make the swap

    The only function that needed overriding is check_improving move. The rest
    of the algorithm remains the same.

    This algorithm may never stop. This is achieved via setting last_improving_iteration
    in iterate, and checking it, and is already a part of the code via the parent class.
    """
    def check_improving_move(self):
        self.current_iter_dist = self.ind.distance
        self.current_iter_sols = []
        i = random.randint(0, self.ind.get_n_cities() - 1)
        for j in range(self.ind.get_n_cities()):
            if not self.ind.is_valid_swap(i, j):
                pass
            else:
                newcost = self.ind.calculate_cost_if_swapped(i, j)
                if newcost == self.current_iter_dist:
                    self.current_iter_sols.append(list(i, j))
                elif newcost < self.current_iter_dist:
                    self.current_iter_dist = newcost
                    self.current_iter_sols = [[i,j]]
                    # We've found an improvement, hence we must return
                    return

student_num = 195734
def main(file_name, n_runs, n_restarts, n_iterations, algorithm, description,\
        use_cache, use_random_heuristic, plot_graph, max_sideways, verbose,\
        allow_sideways, out_file_name):
    if algorithm == 0 or algorithm == '0':
        alg = "base."
    elif algorithm == 1 or algorithm =='1':
        alg = "variant1."
    elif algorithm == 2 or algorithm == '2':
        alg = "variant2."
    else:
        sys.stderr.write(str(algorithm) + str(type(algorithm)))
        assert(False)
    for i in range(n_runs):
        random.seed(student_num + 100 * i)
        output_file_name = out_file_name + alg + str(i+1) + ".csv"
        with open(output_file_name, "w") as f:
            sys.stdout = f
            inst = readInstance(file_name)
            args = {'inst': inst, 'max_sideways_moves':max_sideways,
                    'description': description, 'time_plot': None, 'distance_plot': None,
                    'distance_time_plot': None, 'verbose': verbose,
                    'use_random_heuristic': use_random_heuristic, 'use_cache': use_cache}
            if 0 == algorithm:
                tsp = TSPHillClimbing(**args)
            elif 1 == algorithm:
                tsp = TSPHillClimbingRandomIprovement(**args)
            elif 2 == algorithm:
                tsp = TSPFirstImprovement(**args)
            else:
                assert(False)
            t1 = time.process_time()
            wt1 = time.perf_counter()
            tsp.restart_and_iterate(n_iterations, n_restarts, allow_sideways, max_sideways)
            t1 = time.process_time() - t1
            wt1 = time.perf_counter() - wt1
            f.close()
            output_file_name2 = out_file_name + alg + str(i+1) + ".out"
            with open(output_file_name2, "w") as f:
                sys.stdout = f
                print(f"Best Distance: {tsp.g_best_distance}")
                print(f"Process Time: {t1}")
                print(f"Wall Time: {wt1}")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="Input file name", type=str, required=True)
    parser.add_argument("-nc", "--no-cache", help="Do not use cache", action="store_true")
    parser.add_argument("-n", "--n-runs", help="Number of runs", type=int, required=True)
    parser.add_argument("-a", "--algorithm", help="0 - exhaustive , 1 = random, 2 = first improvement", choices=['0', '1', '2'], default=1)
    parser.add_argument("-r", "--restarts", help="Number of restarts", type=int, required=True)
    parser.add_argument("-i", "--iterations", help="Number of iterations in each restart", type=int, required=True)
    parser.add_argument("-rh", "--random-heuristic", help="Use random heuristic", action="store_true")
    parser.add_argument("-d", "--description", help="Description of the run", type=str, required=False, default="")
    parser.add_argument("-plot", "--plot-graph", help="Plot the graph as well", action="store_true")
    parser.add_argument("-msm", "--max-sideways-moves", help="Maximum number of sideways moves", type=int, default=500)
    parser.add_argument("-v", "--verbose", help="Verbose printing", action="store_true")
    parser.add_argument("-s", "--allow-sideways", help="Allow sideways moves", action="store_true")
    args = parser.parse_args()

    file_name = args.file_name
    n_runs = args.n_runs
    n_restarts = args.restarts
    n_iterations = args.iterations
    algorithm = int(args.algorithm)
    print("algorithm is", algorithm, type(algorithm))
    description = args.description
    use_cache = not args.no_cache
    use_random_heuristic = args.random_heuristic
    plot_graph = args.plot_graph
    max_sideways = args.max_sideways_moves
    verbose = args.verbose
    allow_sideways = args.allow_sideways
    out_file_name = file_name[:-3]
    main(file_name, n_runs, n_restarts, n_iterations, algorithm, description,\
        use_cache, use_random_heuristic, plot_graph, max_sideways, verbose,\
        allow_sideways, out_file_name)
