

"""
Basic TSP Example
file: Individual.py
"""

import collections
import random
import math
import uuid


class Individual:
    def __init__(self, _size, _data, cgenes):
        """
        Parameters and general variables
        """
        self.fitness    = 0
        self.genes      = []
        self.genSize    = _size
        self.data       = _data
        """
        Associate an UUID, needed for debug code
        """
        self.uuid       = uuid.uuid4()

        if cgenes: # Child genes from crossover
            self.genes = cgenes
        else:   # Random initialisation of genes
            self.genes = list(self.data.keys())
            random.shuffle(self.genes)

    def copy(self):
        """
        Creating a copy of an individual
        """
        ind = Individual(self.genSize, self.data,self.genes[0:self.genSize])
        ind.fitness = self.getFitness()
        """
        Copy the UUID, as the sequence is exactly the same
        """
        ind.uuid = self.uuid
        return ind

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return math.sqrt( (d1[0]-d2[0])**2 + (d1[1]-d2[1])**2 )

    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        self.fitness    = self.euclideanDistance(self.genes[0], self.genes[len(self.genes)-1])
        for i in range(0, self.genSize-1):
            self.fitness += self.euclideanDistance(self.genes[i], self.genes[i+1])

    def __eq__(self, other):
        """
        Help check for duplicates, needed for debug code
        """
        assert(None == other or isinstance(other, Individual))
        if None == other:
            return False
        if None == self.genes and None != other.genes:
            return False
        if None != self.data and None == other.genes:
            return False
        if len(self.genes) != len(other.genes):
            return False
        self_keys = self.genes
        other_keys = other.genes
        for i in range(len(self_keys)):
            if self_keys[i] != other_keys[i]:
                return False
        return True

    def validate(self):
        """
        Validate that it is a proper setting
        """
        allkeys = list(self.data.keys())
        for key in allkeys:
            if key not in self.genes:
                return False
        if len(allkeys) != len(self.genes):
            return False
        duplicates = [item for item, count in collections.Counter(self.genes).items() if count > 1]
        #print ("\n", "-" * 80, "\n", allkeys,"\n", self.genes, "\n", duplicates, "\n", "*" * 80)
        return None == duplicates or 0 == len(duplicates)


    def __hash__(self):
        """
        Again, needed for debug code to find duplicates
        """
        return self.uuid.int

    def __repr__(self):
        return f"Individual: {self.uuid.urn}"
