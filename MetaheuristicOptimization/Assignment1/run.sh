mkdir -p RESULTS
for FNAME in inst-0 inst-13 inst-5
do
	FILENAME="TSPData/${FNAME}.tsp"
	
	echo "BASIC GA"
	echo "BASIC GA" > log.txt
	for CONF in 1 2
	do
		RUN_NAME="${FNAME}_conf_${CONF}"
		DIRNAME="./RESULTS/${RUN_NAME}"
		mkdir -p $DIRNAME
		rm -f *.pickle
		rm -f out.txt
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 100 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 100 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt > command.txt
		./TSP_00195734.py --file-name $FILENAME --population-size 100 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt > out.txt
		mv command.txt out.txt *.pickle $DIRNAME
	done

	echo "OTHER CONFIGURATIONS"
	echo "OTHER CONFIGURATIONS" >> log.txt
	for CONF in 3 4 5 6
	do
		RUN_NAME="${FNAME}_conf_${CONF}"
		DIRNAME="./RESULTS/${RUN_NAME}"
		mkdir -p $DIRNAME
		rm -f *.pickle
		rm -f out.txt
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt > command.txt
		./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt > out.txt
		mv command.txt out.txt *.pickle $DIRNAME
	done

	echo "VARY MUTATION RATE"
	echo "VARY MUTATION RATE" >> log.txt
	for CONF in 1 2 3 4 5 6
	do
		RUN_NAME="VARY_MR_${FNAME}_conf_${CONF}"
		DIRNAME="./RESULTS/${RUN_NAME}"
		mkdir -p $DIRNAME
		rm -f *.pickle
		rm -f out.txt
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vmr 0.001 0.01 0.05 0.1 0.5
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vmr 0.001 0.01 0.05 0.1 0.5 > command.txt
		./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vmr 0.001 0.01 0.05 0.1 0.5 > out.txt
		mv command.txt out.txt *.pickle $DIRNAME
	done

	echo "VARY POPULATION SIZE"
	echo "VARY POPULATION SIZE" >> log.txt
	for CONF in 1 2 3 4 5 6
	do
		RUN_NAME="VARY_POPSIZE_${FNAME}_conf_${CONF}"
		DIRNAME="./RESULTS/${RUN_NAME}"
		mkdir -p $DIRNAME
		rm -f *.pickle
		rm -f out.txt
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vps 50 100 500 1000 1500
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vps 50 100 500 1000 1500 > command.txt
		./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vps 50 100 500 1000 1500 > out.txt
		mv command.txt out.txt *.pickle $DIRNAME
	done

	echo "VARY VARY CONFIGURATIONS"
	echo "VARY VARY CONFIGURATIONS" >> log.txt
	for CONF in 1
	do
		RUN_NAME="VARY_CONFIG_${FNAME}_conf_${CONF}"
		DIRNAME="./RESULTS/${RUN_NAME}"
		mkdir -p $DIRNAME
		rm -f *.pickle
		rm -f out.txt
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vc 1 2 3 4 5 6
		echo ./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vc 1 2 3 4 5 6 > command.txt
		./TSP_00195734.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vc 1 2 3 4 5 6 > out.txt
		mv command.txt out.txt *.pickle $DIRNAME
	done
done
