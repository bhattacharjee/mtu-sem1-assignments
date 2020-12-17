
import random
from Queens import *
import matplotlib.pyplot as plt
import sys
import time

studentNum = 195734
random.seed(studentNum)
g_random_seed_counter = 0

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
        return (candidate_sol, self.bCost)

    def solveWithRestarts(self, solve, maxR):
        global g_random_seed_counter
        res = solve()
        self.nRestart = 0
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
        print("Random seed = ", g_random_seed_counter)
        return res

    def solveWithRestartsAndTimeIt(self, solve, maxR):
        self.gStartTime = time.process_time()
        res = self.solveWithRestarts(solve, maxR)
        self.gEndTime = time.process_time()
        self.gRunTime = self.gEndTime - self.gStartTime
        return res

#n, iters, restarts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
n, iters, restarts = 54, 50000, 1000
for i in range(100):
    hc = HillClimbing(n,iters,restarts)
    hc.allow_sideways = True
    hc.no_verbose_print = True
    hc.use_caching = True
    sol = hc.solveWithRestartsAndTimeIt(hc.solveMaxMin, hc.maxRestarts)
    average_stuck_iteration = 0
    if len(hc.stuckHistory) > 0:
        average_stuck_iteration = sum(hc.stuckHistory) / len(hc.stuckHistory)
    print("==RunNo,Restart,Cost,Time,InitialSolutionGenerationTime,Iterations,HeuristicCalls,HeuristicQueenCalls,AvgNumIterationsBeforeLocalMinima")
    print(f"=={i},{hc.nRestart},{sol[1]},{hc.gRunTime},{hc.solution_generation_time},{hc.gIteration},{hc.gHeuristicCostCount},{hc.gHeuristicCostQueenCount},{average_stuck_iteration}")
    print(hc.stuckHistory)
