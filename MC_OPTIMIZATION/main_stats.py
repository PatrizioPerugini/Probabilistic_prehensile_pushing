import numpy as np
#from mc_optimizer import MotionPlanner
#from utils import plot_solution
import sys
import os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from utils.shapes import *
from multipush_poc_pos import MotionPlannerPos
from multipush_poc import MotionPlanner

def main():
    #N_steps = 3#4#7
    steps = [2]#,3,4]#[2, 3, 4]
    num_eps = 2
    successes = 0
    failures = 0
        
    
    T_s = TShape(0.1,0.1,0.03,0.02)
    T_sq = SShape()

    test_shapes = [T_s]#, T_sq]
    #
    errors = []
    meshes = T_s.get_meshes()
    #-0.0431, -0.0148, 0.8004]          
    #    [0.0217, 0.0104, 0.6463]       
    #    [0.0413, 0.0051, 0.3616]
    comparison_dict = {}
    for shape in test_shapes:
        d_step = {}
        for N_steps in steps:
            
            stats = {}
            for i in range(num_eps):
                #start_position = np.array([0.022, 0.01, -0.])
                #goal_position = np.array([-0.04, -0.02, -0.])
                
                #start_position, goal_position =T_s.sample()
                start_position, goal_position =shape.sample()
                
                eps_val = 0.0001
                d_i = {}
                for eps in range(2):
                    eps_i = 0
                    if eps == 1:
                        eps_i = eps_val
                    
                    print("GOING FOR : \n")
                    print(start_position)
                    print(goal_position)
                    print(f"######### eps_i is: {eps_i} \n")
                    #planner = MotionPlanner(N_steps = N_steps , shape = T_s, initial_state = start_position, goal_state = goal_position)
                    planner = MotionPlanner(N_steps = N_steps , shape = shape, initial_state = start_position, goal_state = goal_position, eps_log = eps_i)
                    initial_guess = planner.get_ig()
                    print(initial_guess)
                    planner.callback_func(initial_guess)
                    print("starting optimization")
                    #T, velocities = planner.optimize_motion_cone(N_steps, initial_guess, start_position, goal_position)
                    try:
                        T, velocities, optimized_state, error, iterations = planner.optimize_motion_cone()
                        errors.append(error)

                    #plot_solution(optimized_state ,goal_position ,meshes, it_num = i, N = N_steps)
                    except Exception as e:
                        failures+=1
                        d_i[eps_i] = -1
                        print("Failed")
                        continue
                    #if T == None or error > 1e-2:
                    #    failures+=1
                    #    #continue
                    successes+=1
                   
                    print("trajectory time is: ", T)
                    print("\nvelocities are: ", velocities)
                    print(f"eps_i was: {eps_i} \n")
                    d_i[eps_i] = iterations
                stats[i] = d_i
            d_step[N_steps] = stats
        comparison_dict["shape"] = d_step

    
    
    print("The total number of runs where : ", num_eps, ", the successes where: ", successes, ", and the failures where: ", failures)
    avg_error = np.mean(errors)

    print("the average error is: ", avg_error)

    print(comparison_dict)

    
def get_switches(switch_list):
    #switch_list = [[0,0,1],[0,1,0]]
    
    n_s = 0
    for i in range(len(switch_list)-1):
        idx_0 = np.argmax(switch_list[i])
        idx_1 = np.argmax(switch_list[i+1])
        if idx_0!=idx_1:
            n_s+=1
    return n_s
def get_path_length(path):
    #path = [[0.01,0.01], [0.01,0.01], [0.01,0.01]]
    x_i = [x for x in path[0]]
    y_i = [y for y in path[1]]
    return np.sum(np.sqrt(np.diff(x_i)**2 + np.diff(y_i)**2))

def splitt(li):
    l = []
    for i in range(0,len(li)-2,3):
        sub = []
        for j in range(3):
            sub.append(li[i+j])
        l.append(sub)
    return l

def stats_kl():
    shape = TShape(0.1,0.1,0.03,0.02)
    min_path_list = []
    path_list =[]
    for i in range(30):
        start_position, goal_position =shape.sample()
        planner_kl = MotionPlannerPos(N_steps = 3 , shape = shape, initial_state = start_position, goal_state = goal_position, path_len = 1)
        planner_no_kl = MotionPlannerPos(N_steps = 3 , shape = shape, initial_state = start_position, goal_state = goal_position, path_len = 0)
        try:
            T, sampled_vels, curr_state, of, iters, c = planner_kl.optimize_motion_cone()
            min_path = get_path_length(curr_state)
            min_path_list.append(min_path)

        #plot_solution(optimized_state ,goal_position ,meshes, it_num = i, N = N_steps)
        except Exception as e:
            print("failed with exception: ", e)
            min_path_list.append(-1)
            #continue
    
        try:
            T, sampled_vels, curr_state, of, iters, c = planner_no_kl.optimize_motion_cone()
            path = get_path_length(curr_state)
            path_list.append(path)

        #plot_solution(optimized_state ,goal_position ,meshes, it_num = i, N = N_steps)
        except Exception as e:
            print("failed with exception: ", e)
            path_list.append(-1)
            #continue

    print("out of 10 the switches are: \n")
    print("no min: ", path_list)
    print("\n")
    print("min: ", min_path_list)

def stats_path():
    shape = TShape(0.1,0.1,0.03,0.02)
    kl_list = []
    no_kl_list =[]
    for i in range(10):
        start_position, goal_position =shape.sample()
        planner_path = MotionPlannerPos(N_steps = 3 , shape = shape, initial_state = start_position, goal_state = goal_position, path_len = 1)
        planner_no_path = MotionPlannerPos(N_steps = 3 , shape = shape, initial_state = start_position, goal_state = goal_position, path_len = 0)
        try:
            T, sampled_vels, curr_state, of, iters, c = planner_path.optimize_motion_cone()
            num_switch_c_kl = get_switches(splitt(c))
            kl_list.append(num_switch_c_kl)

        #plot_solution(optimized_state ,goal_position ,meshes, it_num = i, N = N_steps)
        except Exception as e:
            print("failed with exception: ", e)
            kl_list.append(-1)
            #continue
    
        try:
            T, sampled_vels, curr_state, of, iters, c = planner_no_path.optimize_motion_cone()
            num_switch_c_no_kl = get_switches(splitt(c))
            no_kl_list.append(num_switch_c_no_kl)

        #plot_solution(optimized_state ,goal_position ,meshes, it_num = i, N = N_steps)
        except Exception as e:
            print("failed with exception: ", e)
            no_kl_list.append(-1)
            #continue

    print("out of 10 the switches are: \n")
    print("no kl: ", no_kl_list)
    print("\n")
    print("kl: ", kl_list)
    return


if __name__=='__main__':
    #main()
    #li = [0,0,1,0,0,1,1,0,0,0,1,0]
    #
    #print(splitt(li))
    stats_kl()
    #stats_path()