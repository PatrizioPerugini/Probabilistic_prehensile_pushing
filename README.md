# Probabilistic prehensile pushing
`Prehensile pushing` is the task of manipulating grasped objects by pushing them against the environment. 

This repository provides the implementation of the paper Probabilistic_prehensile_pushing [TBD] and our reimplementation of the paper [In-Hand Manipulation Via Motion Cones] used as a baseline.

* `Optimization-based`: in Probabilistic prehensile pushing the aim is to solve an optimization problem in order to find the velocities that can lead us to a specific pose.
* `Sampling-based`: in In-Hand Manipulation Via Motion Cones the solution is found using a sampling algorithm.

# Quick Start

The file structure is shown below

```
Probabilistic_prehensile_pushing/
│
├── main.py                 
│
├── utils/                
│   ├── fast_mc.py        
│   ├── shapes.py          
│   └── utils.py          
│
├── MC_OPTIMIZATION/        
│   ├── main_nlp.py            
│   ├── optimizer.py         
│   ├── multipush_poc.py
|   ├── multipush_poc_pos.py                   
│
└── in_hand_manipulation/   
    ├── main_rrt.py        
    ├── model.py      
    ├── algorithms/       
        ├── planner.py
        └── rrt.py
```
* The optimization algorithm implementation is in the folder `MC_OPTIMIZATION`.
* The sampling algorithm implementation is in the folder `in_hand_manipulation`.
* The motion cones generation is in the folder `utils`.

To install the dependencies, please run the following command in shell. [TBD]
```shell
pip install -r requirements.txt
```
In order to choose which algorithm to use, please select an algorithm.

```shell
python3 main.py -nlp 1
```
In order to set up other hyperparameters check all available arguments, for a detailed description of each one run:

```shell
python3 main.py -h
```
### Optional Arguments

```shell

  -rrt RRT                                Choose 1 to run RRT.

  -step_s STEP_SIZE_RRT                   Define the step size of RRT.

  -eps EPS_GOAL_RRT                       Define the epsilon threshold to determine RRT convergence.

  -num_rp NUM_RANDOM_POINTS               Number of random points the algorithm is evaluated with.
                                          To initialize a specific goal position (gp) and 
                                          start position (sp) choose this parameter to 0.
                                         

  -gp GOAL_POSITION                       Specify the goal position as a comma-separated list 
                                          of numbers, e.g., `1.0,2.0,3.0`.

  -sp START_POSITION                      Specify the start position as a comma-separated list 
                                          of numbers, e.g., `1.0,2.0,3.0`.

  -nlp NLP                                Choose 1 to run NLP.

  -n_steps NUMBER_OF_STEPS                Specify the number of steps.

  -p POSITION                             Set to 1 to optimize position.

  -t TIME                                 Set to 1 to optimize time.

  -le LAMBDA_ENTROPY                      Define entropy cost.

  -lp LAMBDA_PATH                         Define path length cost.

  -lkl LAMBDA_KL                          Define KL divergence cost.

  -t_max MAXIMUM_TIME                     Specify maximum convergence time.

  -shape OBJECT_SHAPE                     Choose either `S` or `T` as object shape.

  -s_all SAMPLE_FROM_ALL_OBJ              Sample from all objects.

  -d_mc DISCRETIZATION0                   Define the granularity when solving nlp. 0 means no discretization.

```
# Running experiments
To run a specific experiment using the provided `main.py` script,  you will need to select and specify several parameters. The following command is an example of how to evaluate an NLP algorithm while optimizing position and time for 20 different poses, with an additional objective of minimizing the overall path length:
```shell
python3 main.py -num_rp 20 -nlp 1 -p 1 -t 1 -lp 0.1
```
### Explanation of parameters
* `-num_rp 20`: This parameter sets the number of randomly sampled poses to 20.

* `-nlp 1`: This flag activates the NLP (Nonlinear Programming) algorithm. Setting it to 1 enables the NLP optimization mode.

* `-p 1`: This enables position optimization. 

* `-t 1`: This enables time optimization. The total trajectory time becomes an optimization variable as well.

* `-lp 0.1`: This sets the weight for path length optimization to 0.1. A lower value (e.g., 0.01) gives less priority to minimizing the overall path length, while higher values will give higher priority.

 ### General usage

 To adjust the experiment, you can modify the parameters as needed. For instance, if you want to optimize only the position and ignore the time, you can run:
 
 ```shell
python3 main.py -num_rp 20 -nlp 1 -p 1 -lp 0.1
```

