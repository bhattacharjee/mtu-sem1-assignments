
import random
from Queens import *
import matplotlib.pyplot as plt
import sys
import time

studentNum = 12345
random.seed(studentNum)

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
        self.gHeuristicCostCount = 0
        self.gHeuristicCostQueenCount = 0
        self.gStartTime = -1
        self.gEndTime = -1
        self.gRunTime = 0
        self.allow_sideways = True
        self.early_stop = True

    def solveMaxMin(self):
        candidate_sol = self.q.generateRandomState(self.size)
        self.bCost = self.q.getHeuristicCost(candidate_sol)
        self.iteration = -1
        print(f"Initial cost = {self.bCost}")

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
                #print(queens_tried_set, candidate, old_val)
                if self.early_stop and not self.allow_sideways and queens_tried_set == set(max_candidate):
                    print(f"Stoppig early, queens_tried = {queens_tried_set}, max_candidate = {set(max_candidate)}")
                    break
            if self.bCost > cost_i:
                self.bCost = cost_i
        return (candidate_sol, self.bCost)

    def solveWithRestarts(self, solve, maxR):
        res = solve()
        self.nRestart = 0
        print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration)
        while self.nRestart < maxR and res[1] > 0:
            random.seed(studentNum + 100 * self.nRestart)
            self.nRestart +=1
            res = solve()
            print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration, self.gIteration)
            print("Time Taken = ", (time.process_time() - self.gStartTime) / (1 if 0 == self.nRestart else self.nRestart))
        print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration, self.gIteration)
        print("Time Taken = ", (time.process_time() - self.gStartTime) / (1 if 0 == self.nRestart else self.nRestart))
        return res

    def solveWithRestartsAndTimeIt(self, solve, maxR):
        self.gStartTime = time.process_time()
        res = self.solveWithRestarts(solve, maxR)
        self.gEndTime = time.process_time()
        self.gRunTime = self.gEndTime - self.gStartTime
        return res

#n, iters, restarts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
n, iters, restarts = 134, 50000, 1
for i in range(10):
    hc = HillClimbing(n,iters,restarts)
    hc.allow_sideways = False
    sol = hc.solveWithRestartsAndTimeIt(hc.solveMaxMin, hc.maxRestarts)
    print("==RunNo,Cost,Time,Iterations,HeuristicCalls,HeuristicQueenCalls")
    print(f"=={i},{sol[1]},{hc.gRunTime},{hc.gIteration},{hc.gHeuristicCostCount},{hc.gHeuristicCostQueenCount}")
