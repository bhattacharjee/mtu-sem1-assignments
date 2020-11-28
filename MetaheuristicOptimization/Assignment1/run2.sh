mkdir -p RESULTS
for FNAME in inst-0 inst-13 inst-5
do
	echo "VARY VARY CONFIGURATIONS"
	echo "VARY VARY CONFIGURATIONS" >> log.txt
	for CONF in 1
	do
		RUN_NAME="VARY_CONFIG_${FNAME}_conf_${CONF}"
		DIRNAME="./RESULTS/${RUN_NAME}"
		mkdir -p $DIRNAME
		rm -f *.pickle
		rm -f out.txt
		echo ./work.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vc 1 2 3 4 5 6
		echo ./work.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vc 1 2 3 4 5 6 > command.txt
		./work.py --file-name $FILENAME --population-size 300 --mutation-rate 0.05 --n-runs 5 --iterations 1000 --configuration ${CONF} --run-name $RUN_NAME -mt -vc 1 2 3 4 5 6 > out.txt
		mv command.txt out.txt *.pickle $DIRNAME
	done
done
