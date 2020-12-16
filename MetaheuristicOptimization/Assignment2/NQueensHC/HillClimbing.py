
import random
from Queens import *
import matplotlib.pyplot as plt
import sys

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

    def solveMaxMin(self):
        candidate_sol = self.q.generateRandomState(self.size)
        self.bCost = self.q.getHeuristicCost(candidate_sol)
        self.iteration = -1

        while self.iteration < self.maxIterations and self.bCost > 0:
            self.gIteration += 1
            self.iteration += 1
            self.cBHistory.append(self.bCost)
            self.cHistory.append (self.q.getHeuristicCost(candidate_sol))
            self.iHistory.append(self.gIteration)

            max_candidate = []
            max_cost = -1
            # Find queen involved in max conflicts
            for cand_i in range(0, self.size):
                cost_i = self.q.getHeuristicCostQueen(candidate_sol, cand_i)
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
                if min_cost > cost_i:
                    min_cost = cost_i
                    best_pos = [pos_i]
                elif min_cost == cost_i and min_cost != self.bCost:
                    # Note this will allow sideways moves
                    # EDIT: Preventing sideways moves with the added condition
                    # min_cost != self.bCost
                    best_pos.append(pos_i)
            if best_pos:
                # Some non-worsening move found
                candidate_sol[candidate] = best_pos[ random.randint(0, len(best_pos)-1) ]
                cost_i = self.q.getHeuristicCost(candidate_sol)
            else:
                # Put back previous sol if no improving solution
                candidate_sol[candidate]=old_val
            if self.bCost > cost_i:
                self.bCost = cost_i
        return (candidate_sol, self.bCost)

    def solveWithRestarts(self, solve, maxR):
        res = solve()
        self.nRestart = 0
        print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration)
        while self.nRestart < maxR and res[1] > 0:
            self.nRestart +=1
            res = solve()
            print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration, self.gIteration)
        print ("Restart: ",self.nRestart, "Cost: ",res[1], "Iter: ",self.iteration, self.gIteration)
        return res

#n, iters, restarts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
n, iters, restarts = 134, 1000, 1
for i in range(1):
    hc = HillClimbing(n,iters,restarts)
    sol = hc.solveWithRestarts(hc.solveMaxMin, hc.maxRestarts)

