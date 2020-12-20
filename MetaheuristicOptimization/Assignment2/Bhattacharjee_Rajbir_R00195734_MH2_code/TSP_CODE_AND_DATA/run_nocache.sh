#!/bin/ksh93

# Usage run_nocache.sh <algorithm>
# where algorithm is either 0, 1 or 2 to indicate
# basic 2opt, variant 1 (random edge) or variant 2 (random edge with first
# improving combination)

#for FILENAME in small/inst-0.tsp
ALG=$1
#for ALG in 2 1 0
#do
	for FILENAME in TSPNoCache/inst-0.tsp TSPNoCache/inst-13.tsp TSPNoCache/inst-5.tsp
	do
		THEDATE=`date`
		echo "Started run for file ${FILENAME} with algorithm ${ALG} on ${THEDATE}"
		./tsp2opt.py --file-name $FILENAME --n-runs 5 --restarts 10 --iterations 10000 --algorithm $ALG --no-cache
		echo "Finished on ${THEDATE}"
	done
#done

