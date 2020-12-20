Environment required:
Any system with python 3, Numpy, Pandas

Note: Files may be either in dos or unix format.

All code and data regarding the TSP solution is in the folder TSP_CODE_AND_DATA.
All code and data regarding the NQueens implementation is in the folder NQUEENS_CODE_AND_DATA.

TSP:
	Python files:
		Individual.py  lab_tsp_insertion.py  tsp2opt.py
		The main file to run is tsp2opt.py
		A help can be obtained by ./tsp2opt.py --help
	Shell Scripts:
		Two driver shell scripts are provided:
			run.sh and run_nocache.sh
				# Usage run.sh <algorithm>
				# where algorithm is either 0, 1 or 2 to indicate
				# basic 2opt, variant 1 (random edge) or variant 2 (random edge with first
				# improving combination)
			These files were run to obtain the data shared here
	Cities data:
		Contained in the folder TSPdata
	Output:
		Contained in the folder: tsp_runs
		tsp_runs contains the following sub-folders:
			1. cache - this contains data from the cached runs
			2. nocache - this contains data from the non-cached runs
			3. plot_charts.ipynb - this is the jupyter notebook used to plot the charts from the output
			There are two type of files inside the folders cache and nocache:
				- *.out - this contains the final value of each
				  of the runs
				- *.csv - this contains data from each iteration
				  of a run

NQUEENS:
	python files:
		HillClimbing.py
			The main file to solve the problem is HillClimbing.py
			This has been modified to take arguments. A help can be obtained by
			typing the command ./HillClimbing.py --help
		Queens.py:
			Helper routine to HillClimbing.py
		convert.py:
			The raw output from HillClimbing.py can be converted to csv
			files using this script
	raw_data:
		Raw data is is in this folder, output of HillClimbing.py
	csv:
		The raw data was converted using conver.py. The converted csvs are in this folder
	Jupyter Notebook:
		plot_rtd.ipynb
		This file was used to plot all the run-time distributions. To re-run this,
		this must be copied to the folder 'csv', and started from there.
	samples.txt:
		Sample file with all the commands that were run

TSP command line arguments:
	$ ./tsp2opt.py --help
	usage: tsp2opt.py [-h] -f FILE_NAME [-nc] -n N_RUNS [-a {0,1,2}] -r RESTARTS
			  -i ITERATIONS [-rh] [-d DESCRIPTION] [-plot]
			  [-msm MAX_SIDEWAYS_MOVES] [-v] [-s]

	optional arguments:
	  -h, --help            show this help message and exit
	  -f FILE_NAME, --file-name FILE_NAME
				Input file name
	  -nc, --no-cache       Do not use cache
	  -n N_RUNS, --n-runs N_RUNS
				Number of runs
	  -a {0,1,2}, --algorithm {0,1,2}
				0 - exhaustive , 1 = random, 2 = first improvement
	  -r RESTARTS, --restarts RESTARTS
				Number of restarts
	  -i ITERATIONS, --iterations ITERATIONS
				Number of iterations in each restart
	  -rh, --random-heuristic
				Use random heuristic
	  -d DESCRIPTION, --description DESCRIPTION
				Description of the run
	  -plot, --plot-graph   Plot the graph as well
	  -msm MAX_SIDEWAYS_MOVES, --max-sideways-moves MAX_SIDEWAYS_MOVES
				Maximum number of sideways moves
	  -v, --verbose         Verbose printing
	  -s, --allow-sideways  Allow sideways moves

NQueens command line arguments:
	$ ./HillClimbing.py --help
	usage: HillClimbing.py [-h] -q N_QUEENS -n N_RUNS -i ITERS -r RESTARTS [-s]
			       [-c] [-v] [-e]

	optional arguments:
	  -h, --help            show this help message and exit
	  -q N_QUEENS, --n-queens N_QUEENS
				Number of queens
	  -n N_RUNS, --n-runs N_RUNS
				Number of runs
	  -i ITERS, --iters ITERS
				Number of iterations
	  -r RESTARTS, --restarts RESTARTS
				Number of restarts
	  -s, --allow-sideways  Allow Sideways moves
	  -c, --use-caching     Use caching for speed-up
	  -v, --verbose-printing
				Verbose printing
	  -e, --early-stopping  Stop early if it is certain no improving mood can be
				found (valid only if sideways is not allowed)
