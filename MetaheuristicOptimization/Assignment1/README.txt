LIST OF FILES
-------------

* TSP_00195734.py
    The main script of the Genetic Algorithm
* Individual.py
    The file containing the Individual class. This was supplied in the
    assignment and I didn't write it.
    Minor changes were made for debugging
* lab_tsp_insertion.py
    Solutions from the lab about for the TSP solution with heuristics and
    random selection. This is used to create the initial population. I didn't
    write this code.
* plotagain.py
    Script to re-render saved matplotlib plots
* plotted.tar.bz2
    Plots from the runs were saved, and zipped. These plots can be re-rendered
    by startin plotagain.py with the pickle file as argument
* sample.txt
    Sample file containing commands that were run

_______________________________________________________________________________

NOTE ON PERFORMANCE
-------------------

During the course of development, several asserts were added for debugging.
These asserts slow down the code by a factor between 2X and 10X.

It is recommended to turn these asserts off by invoking python with the "-OO"
option.

_______________________________________________________________________________

ENVIRONMENT
-----------

This was created in python 3.6.9 and was tested in Windows Subsystem for Linux
(Ubuntu 18.0.4). Additionally, it was verified that the code works on OSX
Catalina 10.15.5.

There are single and multi-processing versions within the same script.
The multi-processing version can be invoked by using -mt option. By default,
it uses 7 CPUs in parallel. That can be changed by changing the global variable
g_n_processes.

There is no single-threaded version of the -vepr option.

When spawning multiple processes, after spawning each process, the code sets
the random seed to student_id + 100 * N. This was a suggestion from
Dr. Grimes and works well with predictable results.


_______________________________________________________________________________

IMPORTED MODULES
----------------

random
collections
time
numpy
matplotlib
statistics
json
pickle
heapq
multiprocessing
math
uuid
_______________________________________________________________________________

USAGE
-----

usage: TSP_00195734.py [-h] -f FILE_NAME [-ps POPULATION_SIZE]
                       [-mr MUTATION_RATE] [-nr N_RUNS] [-ng]
                       [-c {1,2,3,4,5,6}] [-i ITERATIONS]
                       [-vmr [VARY_MUTATION_RATE [VARY_MUTATION_RATE ...]]]
                       [-vps [VARY_POPULATION_SIZE [VARY_POPULATION_SIZE ...]]]
                       [-vc [VARY_CONFIGS [VARY_CONFIGS ...]]]
                       [-vf [VARY_FILES [VARY_FILES ...]]] [-name RUN_NAME]
                       [-mt] [-ucl] [-ocl] [-epr ELITIST_PARENTS_RATIO]
                       [-vepr]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE_NAME, --file-name FILE_NAME
                        File name to parse, str
  -ps POPULATION_SIZE, --population-size POPULATION_SIZE
                        Population Size
  -mr MUTATION_RATE, --mutation-rate MUTATION_RATE
                        Mutation rate
  -nr N_RUNS, --n-runs N_RUNS
                        Number of runs
  -ng, --no-graphs      Do not show any graphs
  -c {1,2,3,4,5,6}, --configuration {1,2,3,4,5,6}
                        Configuration
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations to perform
  -vmr [VARY_MUTATION_RATE [VARY_MUTATION_RATE ...]], --vary-mutation-rate [VARY_MUTATION_RATE [VARY_MUTATION_RATE ...]]
                        Plot with varying mutation rate, specified as a list
  -vps [VARY_POPULATION_SIZE [VARY_POPULATION_SIZE ...]], --vary-population-size [VARY_POPULATION_SIZE [VARY_POPULATION_SIZE ...]]
                        Plot with varying population size, specified as a list
  -vc [VARY_CONFIGS [VARY_CONFIGS ...]], --vary-configs [VARY_CONFIGS [VARY_CONFIGS ...]]
                        Compare different configurations
  -vf [VARY_FILES [VARY_FILES ...]], --vary-files [VARY_FILES [VARY_FILES ...]]
                        Compare across different files (gene size)
  -name RUN_NAME, --run-name RUN_NAME
                        Name of run, matplotlib pickles will be saved with
                        this name
  -mt, --multi-threaded
                        Run multi-threaded versions
  -ucl, --uniform_crossover-large
                        Choose between50 and 75pc of genes for uniform
                        crossover
  -ocl, --order-one-crossover-large
                        Choose between50 and 75pc of genes for uniform
                        crossover
  -epr ELITIST_PARENTS_RATIO, --elitist-parents-ratio ELITIST_PARENTS_RATIO
                        ratio of parents to choose for elitism, between 0 and
                        0.99, negative specifies no elitism
  -vepr, --vary-elitist-parents-ratio
                        Vary elitist parents ratio from, doesn't take any
                        arguments but tries a fixed list of ratios
