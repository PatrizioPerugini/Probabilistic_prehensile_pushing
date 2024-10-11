import numpy as np
import time
import sys
import os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-1]))
from utils.utils import random_points_test
from utils.shapes import *
from multipush_poc_pos import MotionPlannerPos
from multipush_poc import MotionPlanner

def run_NLP(args):
    #N_steps = 3#4#7
    steps = [args.number_of_steps]#[2,3,4,5]#,4]#[2, 3, 4]
    num_eps = 0
    successes = 0
    failures = 0
    start_positions, goal_positions = random_points_test(num_points=args.num_random_points)
    #goal_positions=  [np.array([-0.027, -0.027,0])]
    #start_positions =  [np.array([0.018, 0.005,0])]
    
    shape = None
    if args.object_shape == 'T':
        print("using T-shape")
        shape = TShape(0.1,0.1,0.03,0.02)
    elif args.object_shape == 'S':
        print("using square")
        shape = SShape()
    else:
        raise Exception("No such object")
    #T_s = TShape(0.06,0.06,0.02,0.01)
    
    test_shapes = [shape]#, T_sq]
    errors = []
    meshes = shape.get_meshes()

    #dictionary will be: 
    #{ it_num : {method1 : {num_steps: [num_it, distance, time]}, method1 : {num_steps: [num_it, distance, time]}, method3: [num_it, distance, time}, {it_num :...}   }

    comparison_dict = {}
    for shape in test_shapes:
        d_step = {}
        for N_steps in steps:
            stats = {}
            #for i in range(num_eps):
            for i in range(len(start_positions)):
                planners_stats = {}
                #tests
                start_position = start_positions[i]
                goal_position = goal_positions[i]
                
                if args.sample_from_all_obj == 1:
                    #sample a different goal in the object or from a selected portion
                    start_position, goal_position = shape.sample()
                

                print("GOING FOR : \n")
                print(start_position)
                print(goal_position)

                #which variables are we optimizing?
                
                #position, time, velocity
                if args.position == 1:
                    
                    #optimize velocity, position and time
                    if args.time == 1:
                        print("optimizing pushers, velocity, position and time")
                    else:
                        print("optimizing pushers, velocity and position")

                    planner = MotionPlannerPos(N_steps=N_steps, shape=shape, 
                                                args=args, 
                                                initial_state=start_position, 
                                                goal_state=goal_position)
                    try:
                        start_time = time.time()
                        T, velocities, optimized_state, error, iterations, pushers = planner.optimize_motion_cone()
                        stop_time = time.time()
                        elapsed_time = stop_time - start_time
                        #velocities, optimized_state, error, iterations, pushers = planner.optimize_motion_cone()
                        errors.append(error)
                        planners_stats["MP_VEL_POS"] =  [iterations, error, elapsed_time]
                        #stats[i] =
                        print(optimized_state)
                        successes+=1
                        
                    
                    except Exception as e:
                        failures+=1
                        print("Failed with exception:", e)
                    
                   
                        
                elif args.position == 0:
                    #optimize velocity and time
                    
                    if args.time == 1:
                        print("optimizing pushers, velocity and time")
                    else:
                        print("optimizing pushers and velocity")

                        planner = MotionPlanner(N_steps=N_steps, shape=shape,  
                                                    args=args, 
                                                    initial_state=start_position, 
                                                    goal_state=goal_position)
                        try:
                            
                            start_time = time.time()
                            T, velocities, optimized_state, error, iterations, pushers = planner.optimize_motion_cone()
                            stop_time = time.time()
                            elapsed_time = stop_time - start_time

                            #velocities, optimized_state, error, iterations, pushers = planner.optimize_motion_cone()
                            errors.append(error)
                            planners_stats["MP_VEL"] =  [iterations, error, elapsed_time]
                            #stats[i] =
                            print(optimized_state)
                            successes+=1
                        
                        except Exception as e:
                            failures+=1
                            print("Failed with exception:", e)
                   
                        
                else:
                    raise Exception("Chose the right variables to optimize")




                
                stats[i] = planners_stats
                #print("trajectory time is: ", T)
                #print("\nvelocities are: ", velocities)
                   
                
            d_step[N_steps] = stats
        comparison_dict[shape.name] = d_step

    
    
    print("The total number of runs where : ", num_eps, ", the successes where: ", successes, ", and the failures where: ", failures)
    avg_error = np.mean(errors)

    print("the average error is: ", avg_error)

    print(comparison_dict)
    filename = "results_mindist.pickle"
    #write_results_to_file(comparison_dict,filename= filename)



#if __name__=='__main__':
#    run_NLP()
