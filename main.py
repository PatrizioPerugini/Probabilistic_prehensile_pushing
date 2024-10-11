import argparse
from in_hand_manipulation.main_rrt import run_RRT
from MC_OPTIMIZATION.main_nlp import run_NLP

parser = argparse.ArgumentParser()
parser.add_argument('-rrt', '--RRT', default=1)
parser.add_argument('-step_s', '--step_size_rrt', default=0.05)
parser.add_argument('-eps', '--eps_goal_rrt', default=1e-3)
parser.add_argument('-num_rp', '--num_random_points', default=1)
parser.add_argument('-nlp', '--NLP', default=0)
parser.add_argument('-n_steps', '--number_of_steps', default=3)
parser.add_argument('-p', '--position', default=1)
parser.add_argument('-t', '--time', default=1)
parser.add_argument('-le', '--lambda_entropy', default=0.1)
parser.add_argument('-lp', '--lambda_path', default=0)
parser.add_argument('-lkl', '--lambda_kl', default=0)
parser.add_argument('-t_max', '--maximum_time', default=90)
parser.add_argument('-shape', '--object_shape', default='S' )
parser.add_argument('-s_all', '--sample_from_all_obj', default=0 )
#set it to zero for no discretization or decicde the granularity
parser.add_argument('-d_mc', '--discretization', default=0)



args = parser.parse_args()

if args.NLP == 1:
    print("Running NLP algorithm")
    run_NLP(args)
    

elif args.RRT == 1:
    print("Running RRT algorithm")
    run_RRT(args)

else:
    raise Exception("Choose one method")
