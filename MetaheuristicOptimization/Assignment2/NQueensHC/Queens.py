
import math
import random
from functools import lru_cache

# It is just a helper class , does not actually have any state
# other than the size
class Queens:
    def __init__(self, _size):
        self.size = _size
        self.use_caching = True

    def genereateState(self, _size):
        # Generates an array of 0s
        self.size = _size
        state = [0]*self.size
        return state
    
    def randomState(self, candidate):
        # Overwrite a candidate with a random state
        res = []
        for i in range(0, len(candidate)):
            res.append(random.randint(0, len(candidate)-1))
        return res

    def generateRandomState(self, _size):
        # Overwrite oneself with a random state
        res = self.randomState( self.genereateState(_size) )
        return res

    """
    def getHeuristicCost_nocache(self, candidate):
        # Returns the total number of conflicts
        conflicts = 0
        for index1 in range(0, len(candidate)):
            for index2 in range(index1+1, len(candidate)):
                if( candidate[index1] == candidate[index2]
                        or math.fabs(candidate[index1] - candidate[index2]) == math.fabs(index2 - index1) ):
                    conflicts += 1
        return conflicts
    """

    def getHeuristicCost_nocache(self, candidate):
        # Returns the total number of conflicts
        conflicts = 0
        for index1 in range(0, len(candidate)):
            for index2 in range(index1+1, len(candidate)):
                #if( candidate[index1] == candidate[index2]
                #        or math.fabs(candidate[index1] - candidate[index2]) == math.fabs(index2 - index1) ):
                x1 = candidate[index1] - candidate[index2]
                x2 = index2 - index1
                if( candidate[index1] == candidate[index2] or
                        x1 == x2 or x1 == (-1 * x2)):
                    conflicts += 1
        return conflicts

    @lru_cache(maxsize=256*256*256)
    def getHeuristicCost_lru(self, candidate):
        # Returns the total number of conflicts
        conflicts = 0
        for index1 in range(0, len(candidate)):
            for index2 in range(index1+1, len(candidate)):
                #if( candidate[index1] == candidate[index2]
                #        or math.fabs(candidate[index1] - candidate[index2]) == math.fabs(index2 - index1) ):
                x1 = candidate[index1] - candidate[index2]
                x2 = index2 - index1
                if( candidate[index1] == candidate[index2] or
                        x1 == x2 or x1 == (-1 * x2)):
                    conflicts += 1
        return conflicts

    def getHeuristicCost(self, candidate):
        if self.use_caching:
            return self.getHeuristicCost_lru(tuple(candidate))
        else:
            return self.getHeuristicCost_nocache(candidate)

    """
    def getHeuristicCostQueen(self, candidate, queenId):
        # Return the number of conflicts for a given queen
        conflicts = 0
        for index in range(0, len(candidate)):
            if queenId == index:
                continue
            if ((candidate[queenId] == candidate[index]) or math.fabs(candidate[queenId] - candidate[index]) == math.fabs(index - queenId) ):
                conflicts += 1
        return conflicts
    """
    def getHeuristicCostQueen_nocache(self, candidate, queenId):
        # Return the number of conflicts for a given queen
        conflicts = 0
        for index in range(0, len(candidate)):
            if queenId == index:
                continue
            #if ((candidate[queenId] == candidate[index]) or math.fabs(candidate[queenId] - candidate[index]) == math.fabs(index - queenId) ):
            c1, c2 = candidate[queenId], candidate[index]
            x1 = c1 - c2
            x2 = index - queenId
            if ((c1 == c2) or x1 == x2 or x1 == (-1 * x2)):
                conflicts += 1
        return conflicts

    @lru_cache(maxsize=256*256*256)
    def getHeuristicCostQueen_lru(self, candidate, queenId):
        # Return the number of conflicts for a given queen
        conflicts = 0
        for index in range(0, len(candidate)):
            if queenId == index:
                continue
            #if ((candidate[queenId] == candidate[index]) or math.fabs(candidate[queenId] - candidate[index]) == math.fabs(index - queenId) ):
            c1, c2 = candidate[queenId], candidate[index]
            x1 = c1 - c2
            x2 = index - queenId
            if ((c1 == c2) or x1 == x2 or x1 == (-1 * x2)):
                conflicts += 1
        return conflicts

    def getHeuristicCostQueen(self, candidate, queenId):
        if self.use_caching:
            return self.getHeuristicCostQueen_lru(tuple(candidate), queenId)
        else:
            return self.getHeuristicCostQueen_nocache(candidate, queenId)

    def printSolution(self, candidate):
        # Print the solution
        for i in range(0, len(candidate)):
            var = ""
            for j in range(0, len(candidate)):
                if candidate[i] == j:
                    var+="X"
                else:
                    var+="."
            print (var)
