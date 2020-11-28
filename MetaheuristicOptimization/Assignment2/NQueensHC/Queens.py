
import math
import random

class Queens:
    def __init__(self, _size):
        self.size = _size

    def genereateState(self, _size):
        self.size = _size
        state = [0]*self.size
        return state
    
    def randomState(self, candidate):
        res = []
        for i in range(0, len(candidate)):
            res.append(random.randint(0, len(candidate)-1))
        return res

    def generateRandomState(self, _size):
        res = self.randomState( self.genereateState(_size) )
        return res

    def getHeuristicCost(self, candidate):
        conflicts = 0
        for index1 in range(0, len(candidate)):
            for index2 in range(index1+1, len(candidate)):
                if( candidate[index1] == candidate[index2]
                        or math.fabs(candidate[index1] - candidate[index2]) == math.fabs(index2 - index1) ):
                    conflicts += 1
        return conflicts

    def getHeuristicCostQueen(self, candidate, queenId):
        conflicts = 0
        for index in range(0, len(candidate)):
            if queenId == index:
                continue
            if ((candidate[queenId] == candidate[index]) or math.fabs(candidate[queenId] - candidate[index]) == math.fabs(index - queenId) ):
                conflicts += 1
        return conflicts

    def printSolution(self, candidate):
        for i in range(0, len(candidate)):
            var = ""
            for j in range(0, len(candidate)):
                if candidate[i] == j:
                    var+="X"
                else:
                    var+="."
            print (var)