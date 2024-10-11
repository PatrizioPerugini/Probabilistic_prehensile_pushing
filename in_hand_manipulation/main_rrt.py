import sys
import os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-1]))
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import sympy as sp
from algorithms.rrt import RRT
from model import Model
from utils.utils import plot_moving_square_with_rotation, edit_video, random_points_test
from utils.shapes import*
import time


def run_RRT(args):
    
    results = []
    push = []
    paths = []
    fail_for_time = 0
    list_of_times = []
    thresholds = [args.eps_goal_rrt]
    step_sizes = [args.step_size_rrt]
    
    shape = None
    if args.object_shape == 'T':
        print("using T-shape")
        shape = TShape(0.1,0.1,0.03,0.02)
    elif args.object_shape == 'S':
        print("using square")
        shape = SShape()
    else:
        raise Exception("No such object")
    

    for thr, ss in zip(thresholds,step_sizes):
        list_times_i = []

        initial_states, goal_states = random_points_test(num_points=args.num_random_points)
        
        for gs, ins in zip(goal_states, initial_states):
            print(f"STARTING NEW GOAL PATH WITH GOAL: {gs} AND STARTING FROM: {ins}" )
            m = Model(shape=shape, goal_state=gs, eps_goal = thr)
            planner = RRT(m, maximum_time = args.maximum_time, step_size = ss)
            seed = np.random.randint(0,10**6)
            np.random.seed(seed)
            start_time = time.time()
            goal, plan,node_list = planner.plan(max_nodes=30)
            stop_time = time.time()
            elapsed_time = stop_time - start_time
            if goal:
                results.append(goal)
                list_times_i.append(elapsed_time)
                pushers = [ m.wall_orientations[node.from_rs] for node in plan if node.from_rs is not None ]
                path = [(node.state[0], node.state[1]) for node in plan]
                push.append(pushers)
                paths.append(path)
                print("Time taken by the planner is: ", elapsed_time )


            elif not goal and node_list == True:
                #Maximum time reached
                list_times_i.append(args.maximum_time)
                results.append(0)
                fail_for_time += 1
            else:
                results.append(0)
                list_times_i.append(args.maximum_time)
        list_of_times.append(list_times_i)
        print(list_of_times)



    print("the list of times is: ",list_of_times)
    print("goals where found if 1: \n",results)
    print(f"Maximum time reached: {fail_for_time} times out of {args.num_random_points}")
    [print("avg is: ", np.mean(list_of_times[i])) for i in range(len(list_of_times))]


#if __name__=="__main__":
#    run_RRT()
