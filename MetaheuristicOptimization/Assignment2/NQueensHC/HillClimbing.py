#!/usr/bin/env python3
import random
from Queens import *
import matplotlib.pyplot as plt
import sys
import time
import argparse

studentNum = 195734
random.seed(studentNum)
g_random_seed_counter = 5

class HillClimbing:
    def __init__(self, _size, _maxIterations, _maxRestarts):
        self.bCost = 0
        self.maxIterations = _maxIterations
        self.maxRestarts = _maxRestarts
        self.gIteration = 0
        self.nRestart = 0
        self.iteration = 0
        self.size = _size
        self.q = Queens(self.size)
        self.bCost = -1
        self.cHistory = []
        self.cBHistory = []
        self.iHistory = []
        self.stuckHistory = [] # After how many iterations did we get stuck?
        self.gHeuristicCostCount = 0
        self.gHeuristicCostQueenCount = 0
        self.gStartTime = -1
        self.gEndTime = -1
        self.gRunTime = 0
        self.allow_sideways = True
        self.early_stop = True
        self.no_verbose_print = False
        self.use_caching = True
        self.solution_generation_time = 0

    def printAgain(self, candidate_sol):
        cost_arr = []
        for cand_i in range(0, self.size):
            cost_arr.append(self.q.getHeuristicCostQueen(candidate_sol, cand_i))
        print("Again: cost = ", self.q.getHeuristicCost(candidate_sol))
        print(f"FINAL COSTS: {cost_arr}")


    def solveMaxMin(self):
        t_sol_gen = time.process_time()
        candidate_sol = self.q.generateRandomState(self.size)
        t_sol_gen = time.process_time() - t_sol_gen
        self.solution_generation_time += t_sol_gen

        self.bCost = self.q.getHeuristicCost(candidate_sol)
        self.iteration = -1
        #print(f"Initial cost = {self.bCost}")
        self.q.use_caching = self.use_caching

        queens_tried_set = set()
        while self.iteration < self.maxIterations and self.bCost > 0:
            self.gIteration += 1
            self.iteration += 1
            #self.cBHistory.append(self.bCost)
            #self.cHistory.append (self.q.getHeuristicCost(candidate_sol))
            self.gHeuristicCostCount += 1
            #self.iHistory.append(self.gIteration)

            max_candidate = []
            max_cost = -1
            # Find queen involved in max conflicts
            for cand_i in range(0, self.size):
                cost_i = self.q.getHeuristicCostQueen(candidate_sol, cand_i)
                self.gHeuristicCostQueenCount += 1
                if max_cost < cost_i:
                    max_cost = cost_i
                    max_candidate = [cand_i]
                elif max_cost == cost_i:
                    # Ties
                    max_candidate.append(cand_i)

            # max_candidate is the list of columns which are candidates
            # row in which the queen is is not stored in this list
            if max_cost == -1:
                break
            # candidate contains the column in which the queen is
            candidate = max_candidate[ random.randint(0, len(max_candidate)-1) ]
            if self.early_stop and not self.allow_sideways:
                queens_tried_set.add(candidate)
            # old_val now contains the row in which the queen was before we started moving it around
            old_val = candidate_sol[candidate]

            ##best move for the selected queen
            min_cost = max_cost
            start_min_cost = min_cost
            best_pos = []

            # Loop through all the rows loooking for a new place for the candidate queen
            for pos_i in range(0, self.size):
                if pos_i == old_val:
                    # Neighbor must be different to current
                    continue
                candidate_sol[candidate] = pos_i
                cost_i = self.q.getHeuristicCostQueen(candidate_sol, candidate)
                self.gHeuristicCostQueenCount += 1
                if min_cost > cost_i:
                    min_cost = cost_i
                    best_pos = [pos_i]
                elif min_cost == cost_i and True == self.allow_sideways:
                    # Note this will allow sideways moves
                    best_pos.append(pos_i)
                elif min_cost == cost_i and min_cost < start_min_cost and False == self.allow_sideways:
                    best_pos.append(pos_i)
            if best_pos:
                # Some non-worsening move found
                candidate_sol[candidate] = best_pos[ random.randint(0, len(best_pos)-1) ]
                cost_i = self.q.getHeuristicCost(candidate_sol)
                self.gHeuristicCostCount += 1
                if self.early_stop and not self.allow_sideways:
                    queens_tried_set.clear()
            else:
                # Put back previous sol if no improving solution
                candidate_sol[candidate]=old_val
                # We may have set cost_i to the cost of an individual queen
                # rather than all queens put together, returning the former
                # would be incorrect
                cost_i = self.q.getHeuristicCost(candidate_sol)
                self.gHeuristicCostCount += 1
                #print(queens_tried_set, candidate, old_val)
                if self.early_stop and not self.allow_sideways and queens_tried_set == set(max_candidate):
                    #print(f"Stoppig early, queens_tried = {queens_tried_set}, max_candidate = {set(max_candidate)}")
                    #self.printAgain(candidate_sol)
                    self.stuckHistory.append(self.iteration)
                    break
            if self.bCost > cost_i:
                self.bCost = cost_i
                if not self.no_verbose_print:
                    print(f"Improving move found in iteration {self.iteration}")
        return (candidate_sol, self.bCost)

    def solveWithRestarts(self, solve, maxR):
        global g_random_seed_counter
        g_random_seed_counter += 1
        random.seed(studentNum + 100 * g_random_seed_counter)
        res = solve()
        self.nRestart = 0
        if not self.no_verbose_print:
            print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration)
        while self.nRestart < maxR and res[1] > 0:
            g_random_seed_counter += 1
            random.seed(studentNum + 100 * g_random_seed_counter)
            self.nRestart +=1
            res = solve()
            if not self.no_verbose_print:
                print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration, self.gIteration)
                print("Time Taken = ", (time.process_time() - self.gStartTime) / (1 if 0 == self.nRestart else self.nRestart))
                print("random_seed_counter = ", g_random_seed_counter)
        print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration, self.gIteration)
        print("Time Taken = ", (time.process_time() - self.gStartTime) / (1 if 0 == self.nRestart else self.nRestart))
        #print("Random seed = ", g_random_seed_counter)
        return res

    def solveWithRestartsAndTimeIt(self, solve, maxR):
        self.gStartTime = time.process_time()
        res = self.solveWithRestarts(solve, maxR)
        self.gEndTime = time.process_time()
        self.gRunTime = self.gEndTime - self.gStartTime
        return res

#n, iters, restarts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
def main(nruns:int, n:int, iters:int, restarts:int, allowSideways:bool, useCaching:bool, verbosePrinting:bool, early_stopping:bool)->None:
    global studentNum
    #n, iters, restarts = 54, 50000, 1000
    run_times = []
    wall_times = []
    for j in range(nruns):
        random.seed(studentNum + 100 * j)
        t1 = time.perf_counter()
        hc = HillClimbing(n,iters,restarts)
        hc.allow_sideways = allowSideways
        hc.early_stop = early_stopping
        hc.no_verbose_print = (not verbosePrinting)
        hc.use_caching = useCaching
        sol = hc.solveWithRestartsAndTimeIt(hc.solveMaxMin, hc.maxRestarts)
        average_stuck_iteration = 0
        if len(hc.stuckHistory) > 0:
            average_stuck_iteration = sum(hc.stuckHistory) / len(hc.stuckHistory)
        print("==RunNo,Restart,Cost,Time,InitialSolutionGenerationTime,Iterations,HeuristicCalls,HeuristicQueenCalls,AvgNumIterationsBeforeLocalMinima")
        print(f"=={j},{hc.nRestart},{sol[1]},{hc.gRunTime},{hc.solution_generation_time},{hc.gIteration},{hc.gHeuristicCostCount},{hc.gHeuristicCostQueenCount},{average_stuck_iteration}")
        print(hc.stuckHistory)
        run_times.append(hc.gRunTime)
        t1 = time.perf_counter() - t1
        wall_times.append(t1)
        print(f"Average Run Time so far in iteration {j}: {sum(run_times) / len(run_times)} {sum(wall_times) / len(wall_times)}")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--n-queens", help="Number of queens", type=int, default=-1, required=True)
    parser.add_argument("-n", "--n-runs", help="Number of runs", type=int, default=-1, required=True)
    parser.add_argument("-i", "--iters", help="Number of iterations", type=int, default=-1, required=True)
    parser.add_argument("-r", "--restarts", help="Number of restarts", type=int, default=-1, required=True)
    parser.add_argument("-s", "--allow-sideways", help="Allow Sideways moves", action="store_true")
    parser.add_argument("-c", "--use-caching", help="Use caching for speed-up", action="store_true")
    parser.add_argument("-v", "--verbose-printing", help="Verbose printing", action="store_true")
    parser.add_argument("-e", "--early-stopping", help="Stop early if it is certain no improving mood can be found (valid only if sideways is not allowed)", action="store_true")
    args = parser.parse_args()

    n = args.n_queens
    nruns = args.n_runs
    iters = args.iters
    restarts = args.restarts
    allow_sideways = args.allow_sideways
    use_caching = args.use_caching
    verbose_printing = args.verbose_printing
    early_stopping = args.early_stopping

    main(nruns, n, iters, restarts, allow_sideways, use_caching, verbose_printing, early_stopping)
