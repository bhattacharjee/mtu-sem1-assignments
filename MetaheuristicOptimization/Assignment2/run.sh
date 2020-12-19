#!/bin/ksh93

#for FILENAME in small/inst-0.tsp
ALG=$1
#for ALG in 2 1 0
#do
	for FILENAME in TSPData/inst-0.tsp TSPData/inst-13.tsp TSPData/inst-5.tsp
	do
		THEDATE=`date`
		echo "Started run for file ${FILENAME} with algorithm ${ALG} on ${THEDATE}"
		./A2.py --file-name $FILENAME --n-runs 5 --restarts 10 --iterations 10000 --algorithm $ALG
		echo "Finished on ${THEDATE}"
	done
#done

