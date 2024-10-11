import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import sympy as sp
import math
from scipy.optimize import minimize
import cProfile
import random
import sys
import os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from utils.utils import *
from utils.fast_mc import MotionCone
from optimizer import Optimizer

np.set_printoptions(precision=4,linewidth=np.inf, suppress=True)

class MotionPlanner(Optimizer):
    def __init__(self, N_steps, shape, args, m = 500/1000, g=9.81, 
                initial_state=np.array([0,0,0]), 
                goal_state=np.array([0.2,0.3,-0.0]), 
                wall_orientations = [0,-np.pi/2, np.pi/2],
                r = 1/100, mu_s = 0.8,
                N_gripper = 50, mu = 0.8, d = 3/100, w = 6/100, eps_log = 0, kl = 0 ):
                
        super().__init__(N_steps, shape, args, m, g, 
                         initial_state, 
                         goal_state, 
                         wall_orientations ,
                         r, mu_s ,
                         N_gripper , mu , d , w )
        
        self.eps_log = eps_log
        self.kl = kl


    def compute_positions(self, velocities_x, velocities_z, angular_velocities_w, T):
        
        dt = T/ (self.N_steps-1)
        p_n_x, p_n_z, p_n_w = self.initial_state
        x_positions = [p_n_x]
        z_positions = [p_n_z]
        w_positions = [p_n_w]
        for vx, vz, vw in zip(velocities_x, velocities_z, angular_velocities_w):
   
            update = box_plus(p_n_x, p_n_z, p_n_w, vx * dt, vz * dt, vw * dt)
            p_n_x, p_n_z, p_n_w = update.flatten()
            x_positions.append(p_n_x)
            z_positions.append(p_n_z)
            w_positions.append(p_n_w)

        return x_positions, z_positions, w_positions


    def compute_positions_probabilities(self, velocities_x, velocities_z, velocities_w, T,c):
        dt = T/ (self.N_steps-1)
        #print("the len of vx is:", len(velocities_x))
        p_n_x, p_n_z, p_n_w = self.initial_state
        x_positions = [p_n_x]
        z_positions = [p_n_z]
        w_positions = [p_n_w]

        for t in range(self.N_steps):
            vx, vz, vw = 0, 0, 0
            #sume over all combinations velocity pusher
            for j in range(self.N_pushers):
                vx += velocities_x[t*self.N_pushers+j]*c[t*self.N_pushers+j]
                vz += velocities_z[t*self.N_pushers+j]*c[t*self.N_pushers+j]
                vw += velocities_w[t*self.N_pushers+j]*c[t*self.N_pushers+j]
            update = box_plus(p_n_x, p_n_z, p_n_w, vx * dt, vz * dt, vw * dt)
            p_n_x, p_n_z, p_n_w = update.flatten()
            x_positions.append(p_n_x)
            z_positions.append(p_n_z)
            w_positions.append(p_n_w)

        return x_positions, z_positions, w_positions


    def objective_function(self, vars):
        T = vars[0]  # Total time
        #vx for each pusher and for each timestep
        velocities_x = vars[1:self.N_pushers*self.N_steps+1]  # Gripper velocities for x
        #velocities_z = vars[self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
        velocities_z = vars[self.N_pushers*self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
        angular_velocities_w = vars[2*self.N_pushers*self.N_steps+1:3*self.N_pushers*self.N_steps+1]  # Angular velocities for w
        #be aware that here the velocities are divided but each velocity must be applied with the pusher probability
        c = vars[3*self.N_pushers*self.N_steps+1:]
        

        p_x, p_z, p_w = self.compute_positions_probabilities(velocities_x, velocities_z, angular_velocities_w, T,c)
        p_n_x, p_n_z, p_n_w = p_x[-1], p_z[-1], p_w[-1]
        dist_xz = np.linalg.norm(np.array([p_n_x, p_n_z]) - self.goal_state[0:2])
        dist_w = np.linalg.norm(np.array([p_n_w]) - self.goal_state[2])
        dist = 0.1*dist_w + 0.9*dist_xz

        entropy_cost = sum(c[t] * np.log(c[t]) for t in range(self.N_steps * self.N_pushers) if c[t] != 0)

        #lambda_e = 0.1#0.05#0.005
        kl_term = np.sum(self.kl_divergence(c))

        return   (1-self.lambda_e)*dist - self.lambda_e*entropy_cost - self.lambda_kl*kl_term

    def eq_constraints(self, vars):
        
        c = vars[3*self.N_pushers*self.N_steps+1:]
        constraints = []
        c_m = 0
        for n in range(self.N_steps*self.N_pushers):
            if n%self.N_pushers==0 and n!=0:
                constraints.append(c_m -1)
                c_m = 0
            c_m+=c[n]
            if n==self.N_steps*self.N_pushers -1:
                constraints.append(c_m -1)

        return np.array(constraints)
    

    def ineq_constraints(self, vars):
        T = vars[0]  # Total time
        velocities_x = vars[1:self.N_pushers*self.N_steps+1]  # Gripper velocities for x
        velocities_z = vars[self.N_pushers*self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
        velocities_w = vars[2*self.N_pushers*self.N_steps+1:3*self.N_pushers*self.N_steps+1]  # Angular velocities for w
        #be aware that here the velocities are divided but each velocity must be applied with the pusher probability
        c = vars[3*self.N_pushers*self.N_steps+1:]
 
        inequalities = []
        twists = []
        for vx, vz, vw in zip(velocities_x,velocities_z,velocities_w):
            current_twist = [vx, vz, vw]
            twists.append(current_twist)
        for _ in range(self.N_pushers):
            twists.append([0, 0, 0])

        p_x, p_z, p_w = self.compute_positions_probabilities(velocities_x, velocities_z, velocities_w, T, c)

        for i, ( x, z, w)  in enumerate(zip(p_x, p_z, p_w)):#, velocities_x, velocities_z, velocities_w):
            current_state = np.array([x, z, w])

            # SQUARE INEQUALITIES
            #inequalities = self.get_square_const(inequalities)
       
            # T-SHAPE INEQUALITIES -> dim of t are: 4 and 10 for bases 3 and 7 for height (incavata in 10x10 square)
            #inequalities = self.get_t_constraints(x, z, inequalities)
            #STO QUA IL PRIMO FUNZIA
            inequalities = self.shape.get_constraints(x, z, inequalities)
          
        # Compute motion cones for the given state and orientations of the wall.
            for j,theta_rad in enumerate(self.wall_orientations):

                motion_cones_i = None
                if self.discretization:
                    motion_cones_i = self.retrive_precomp_mc(wall_orientation = theta_rad, target_state = current_state)
                else:
                    motion_cones_i, _ = self.get_motionCones(current_state, theta_rad = theta_rad)  
                             
                if motion_cones_i == None:
                    for _ in range(6):
                        inequalities.append(0)
                    continue
                    motion_cones_i = [0,0,0]
                

                #motion_cones_i  = self.poc_ch(theta_rad)#ConvexHull(np.array([[1,0,1],[1,-1,1],[-1,1,0],[-1,-1,-1]]))
                
                #idx = print((i-1)*(self.N_pushers)+j)
                #print((i)*(self.N_pushers)+j)

                current_twist = np.array(twists[(i)*(self.N_pushers)+j])
                #print("current twist", current_twist)
                #cnt = 0

                for equation in motion_cones_i.equations:

                    a = equation[:-1]  # Coefficients a1, a2, ..., an
                    b = equation[-1]   # Constant term b

                    # Constructing inequality: a1*x1 + a2*x2 + ... + an*xn + b <= 0
                    #inequality = {"type": "ineq", "fun": lambda x, a=a, b=b: np.dot(a, x) - b}
                    inequality = self.compute_hp(a, b, current_twist)
                    inequalities.append(inequality)

        c_matrix = np.array(c).reshape((self.N_steps, self.N_pushers))
    
        flattened_c = c_matrix.flatten()
    
        for c_ij in flattened_c:
            inequalities.append(c_ij-self.eps_log)  

        return inequalities


    def callback_func(self, xk):
        T = xk[0]
        dt = T/(self.N_steps-1)
        v_x = xk[1:self.N_pushers*self.N_steps+1]  # Gripper velocities for x
        #v_z = xk[self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
        v_z = xk[self.N_pushers*self.N_steps+1:2*self.N_steps*self.N_pushers+1]
        v_w = xk[2*self.N_pushers*self.N_steps+1:3*self.N_pushers*self.N_steps+1]  # Angular velocities for w
        #be aware that here the velocities are divided but each velocity must be applied with the pusher probability
        c = xk[3*self.N_pushers*self.N_steps+1:]

        self.sols = xk
        #get list of positions
        vx, vz, vw = self.retrieve_optim_vels(v_x,v_z,v_w,c)

        v_i = [[vx_,vz_,vw_] for vx_,vz_,vw_  in zip(vx,vz,vw)]

        #px,py,pz = self.compute_positions(v_x, v_z, v_w, self.Traj_time)
        px,py,pz = self.compute_positions_probabilities(v_x, v_z, v_w, T,c)
        curr_state = [[x,z,w] for x,z,w in zip(px,py,pz)]
        self.states_list.append(curr_state)
        print(f"\nIter. {self.callback_iteration}: OF = {self.objective_function(xk)}  -  dt = {T/(self.N_steps-1)}")
        print("{: <10} {: <30} {: <30}".format('TIMESTEP', 'POSITIONS', 'VELOCITIES'))
        t = 0
        for v,p in zip(v_i,curr_state):
            print("{: <10} {: <30} {: <30}".format(t, str([round(i,4) for i in p]), str([round(i,4) for i in v])))
            t += 1
        print("{: <10} {: <30} {: <30}".format(t, str([round(i,4) for i in curr_state[-1]]), str([0., 0., 0.])))
        print(c)

        if self.objective_function(xk) < 0.1:
            self.obj_reached = True

        self.callback_iteration += 1


    def vels_ig(self, T_ig):
        #dist_per_int = (self.goal_state-self.initial_state)/(self.N_steps+1)
        total_vels = (self.goal_state-self.initial_state)/T_ig
        #dist_per_int = (self.goal_state-self.initial_state)/(self.N_steps+1)*(self.N_pushers)



        delta_v = (total_vels*(self.N_steps-1))/self.N_pushers
        ddv = delta_v#/(self.N_pushers-2)
        #print(ddv)
        return ddv



    def get_ig(self):
        T_ig = 0.1
        ddv = self.vels_ig(T_ig)
        initial_guess = [T_ig] + [ddv[0]]*self.N_steps*self.N_pushers + [ddv[1]]*self.N_steps*self.N_pushers + [ddv[2]]*self.N_steps*self.N_pushers +  [1/self.N_pushers]*self.N_steps*self.N_pushers
        #initial_guess = [T_ig] + [-0.] *3*self.N_steps*self.N_pushers + [1/self.N_pushers]*self.N_steps*self.N_pushers
        self.compute_positions_probabilities(initial_guess[1:self.N_steps*self.N_pushers+1], initial_guess[self.N_steps*self.N_pushers+1:2*self.N_steps*self.N_pushers+1], initial_guess[2*self.N_steps*self.N_pushers+1:3*self.N_steps*self.N_pushers+1], initial_guess[0],initial_guess[3*self.N_steps*self.N_pushers+1:])


        return  initial_guess


    def return_solution(self, result):
        T = result[0]
        velocities = result[1:3*self.N_pushers*self.N_steps+1]
        pushers =  result[3*self.N_pushers*self.N_steps+1:]
        v_x = result[1:self.N_pushers*self.N_steps+1]  # Gripper velocities for x
        #v_z = result[self.N_steps+1:2*self.N_steps*self.N_pushers+1]
        v_z = result[self.N_pushers*self.N_steps+1:2*self.N_steps*self.N_pushers+1]  # Gripper velocities for z
        v_w = result[2*self.N_pushers*self.N_steps+1:3*self.N_pushers*self.N_steps+1]
        c = result[3*self.N_pushers*self.N_steps+1:]
        print("Optimal Time T:",T)
        print("velocities:", velocities )
        print("pushers:", pushers)
        vx, vz, vw = self.retrieve_optim_vels(v_x,v_z,v_w,pushers)
        sampled_vels =  [[vx_,vz_,vw_] for vx_,vz_,vw_ in zip(vx,vz,vw) ] 
        px,py,pz = self.compute_positions(vx, vz, vw, T) #here is correct to use it
        curr_state = [[x,z,w] for x,z,w in zip(px,py,pz)]
        of = self.objective_function(result)
        write_results_to_file(self.states_list, "plot_t_evolution.pickle")

        return T, sampled_vels, curr_state, of, self.callback_iteration, c
   
    #velocities = [v_x_0, v_x_1, v]
    def optimize_motion_cone(self):
        
        #bouns on: T + 3NM + NM
        #bounds = [(0.01, 30)] + [(-2, 2)] * (3*self.N_pushers*self.N_steps) + [(0, 1)] * (self.N_steps*self.N_pushers)  # Bounds for T and velocities

        T_min = .08 if self.args.time else self.T_ig
        T_max = 2.8 if self.args.time else self.T_ig
        
        bounds = [(T_min, T_max)] + [(-2, 2)] * (3*self.N_pushers*self.N_steps) + [(0, 1)] * (self.N_steps*self.N_pushers)

        initial_guess = self.get_ig()

        constraints = [{'type': 'eq', 'fun': self.eq_constraints},
                   {'type': 'ineq', 'fun': self.ineq_constraints}]
        options = {'maxiter': 100, 'disp': True, 'ftol':1e-7}#1e-7

 
        result = minimize(self.objective_function, initial_guess, constraints=constraints, bounds=bounds, options=options,
                            callback=self.callback_func, method='SLSQP')
                    
                                                                #trust-constr
        if self.obj_reached or result.success: #result.success:
            T, sampled_vels, curr_state, of, iters, c = self.return_solution(result.x)
            
            if T == None:
                T, sampled_vels, curr_state, of, iters, c = self.return_solution(self.sols)
                return T, sampled_vels, curr_state, of, iters, c

            return T, sampled_vels, curr_state, of, iters, c
        else:
            print("Optimization failed.")
            return None, None, None, None, None, None

