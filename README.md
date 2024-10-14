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
