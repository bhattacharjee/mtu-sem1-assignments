#!/bin/ksh93

#for FILENAME in small/inst-0.tsp
for FILENAME in TSPData/inst-0.tsp TSPData/inst-13.tsp TSPData/inst-5.tsp
do
	for ALG in 0 1 2
	do
		./A2.py --file-name $FILENAME --n-runs 5 --restarts 10 --iterations 10000 --algorithm 0
	done
done

