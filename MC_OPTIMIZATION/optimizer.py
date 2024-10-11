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

np.set_printoptions(precision=4,linewidth=np.inf, suppress=True)

class Optimizer:
    def __init__(self, N_steps, shape, args, m = 500/1000, g=9.81, 
                initial_state=np.array([0,0,0]), 
                goal_state=np.array([0.2,0.3,-0.0]), 
                wall_orientations = [0,-np.pi/2, np.pi/2],
                r = 1/100, mu_s = 0.8,
                N_gripper = 50, mu = 0.8, d = 3/100, w = 6/100):
        # Initialize parameters
        self.m = m
        self.g = g
        self.args = args
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.callback_iteration = 0
        self.obj_reached = False
        self.r = r
        self.rc_q = (0.6 * self.r) ** 2
        self.m = m
        self.states_list = []
        self.mu_s = mu_s
        self.N_gripper= N_gripper
        self.mu = mu
        self.theta_mu =math.atan(self.mu)
        self.N_steps = N_steps #number of  steps
        self.cnt=0
        self.c = math.cos(self.theta_mu)
        self.s = math.sin(self.theta_mu)
        self.d = d  
        self.w = w
        self.sols = None
        self.Traj_time = 0.1
        self.optimize_time = self.args.time
        self.mcones = {}
        self.T_ig = 0.15

        self.MC_solver = MotionCone()
        self.MC_solver.get_functions()
 
        self.wall_orientations = wall_orientations
        self.N_pushers = len(self.wall_orientations)
        self.discretization = self.args.discretization
        
        self.shape = shape
        self.initialize_parameters()
        self.precompute_mc()
        
        self.lambda_kl = self.args.lambda_kl         
        self.lambda_e = self.args.lambda_entropy
        self.lambda_trajectory = self.args.lambda_path
   
    def initialize_parameters(self):
        
        self.state_x = np.linspace(-self.d,self.d,self.discretization)
        self.state_y = np.linspace(-self.d+0.01,self.d-0.01,self.discretization)
      
        self.a = -1/(self.mu_s*self.N_gripper)
        self.b = self.m*self.g*self.a

        self.B = np.diag([1,1,1/self.rc_q])
        
        self.J_p1 = np.array([[1, 0], [0, 1], [-self.w/2, self.d]])  #CONSIDER TO USE THE VALUES OF THE SHAPE DIRECTLY
        self.J_p2 = np.array([[1, 0], [0, 1], [-self.w/2, -self.d]]) #CONSIDER TO USE THE VALUES OF THE SHAPE DIRECTLY


        self.f_r = [np.cos(np.pi/2-math.atan(self.mu)),  np.sin(np.pi/2-math.atan(self.mu))]
        self.f_l = [-np.cos(np.pi/2-math.atan(self.mu)), np.sin(np.pi/2-math.atan(self.mu))]

    def apply_jacobian_rotation(self, theta_rad):


        wp1_r = np.dot(self.J_p1, self.f_r)
        wp1_l = np.dot(self.J_p1, self.f_l)
        wp2_r = np.dot(self.J_p2, self.f_r)
        wp2_l = np.dot(self.J_p2, self.f_l)
        
        return wp1_r, wp1_l, wp2_r, wp2_l
        
    def compute_rigid_transform(self, x):
        c = math.cos(x[2])
        s = math.sin(x[2])
        tx = x[0]
        tz = x[1]
        T = np.array([[c, -s, tx], [s, c, tz], [0, 0, 1]])
        return T

    def get_motionCones(self, x, theta_rad, vel_i=None):
        num_sol = 4
        J_s = self.compute_rigid_transform(x)

        wp1_r, wp1_l, wp2_r, wp2_l = self.apply_jacobian_rotation(theta_rad)
        wrench_cone = np.vstack([wp1_r, wp1_l, wp2_r, wp2_l])

        sol = []
        for edge_wrench in wrench_cone:
   
            c   = math.cos(x[2])
            s   = math.cos(x[2])
            t_x = x[0]
            t_z = x[1]

            a1 = edge_wrench[0]*self.a
            a2 = edge_wrench[1]*self.a
            a3 = edge_wrench[2]*self.a

            F_x_val, F_z_val, M_y_val, K_val = self.MC_solver.retrieve_solution(c,s,t_x,t_z,a1,a2,a3,self.b,self.rc_q)

            solution = np.array([F_x_val, F_z_val, M_y_val, K_val])
            sol.append(solution)
        solutions = np.array(sol)
        
        ws_h = []
        for i in range(len(sol)):
            ws_h.append(solutions[i][0:num_sol-1])
        v_obj = []
        for ws in ws_h:
            ws = np.reshape(ws, (3, 1))
            v_i = np.linalg.inv(J_s) @ self.B @ (ws)
            v_obj.append(v_i)
        V_obj = np.reshape(np.array(v_obj,dtype=float), (num_sol, 3,))
        V_origin = np.array([0, 0, 0])

        check_hull_vel = np.vstack([V_obj[0][:], V_obj[1][:], V_obj[2], V_obj[3], V_origin])
        if np.isnan(check_hull_vel).any():
            return None, None
        else:
            axis = [0,0,1]
            rotated_cone_points = rotate_cone(check_hull_vel, axis, theta_rad)
            hull = ConvexHull(rotated_cone_points)

            return hull, check_hull_vel

    #Discretize the state space 
    def precompute_mc(self):
        
        for w_o in self.wall_orientations:
            self.mcones[w_o] = {}
            for s_x in self.state_x:
                self.mcones[w_o][s_x] = {}
                for s_y in self.state_y:
                    state = [s_x, s_y, 0]
                    mci, _ = self.get_motionCones(state,w_o)
                    self.mcones[w_o][s_x][s_y] = mci


        
    #Discretize the state space, utils to recover the closest
    def retrive_precomp_mc(self, wall_orientation, target_state):
        
        target_x, target_y, _ = target_state
        
        closest_s_x = min(self.state_x, key=lambda s_x: abs(s_x - target_x))
        closest_s_y = min(self.state_y, key=lambda s_y: abs(s_y - target_y))
        return self.mcones[wall_orientation][closest_s_x][closest_s_y]


    def kl_divergence(self, c):
        kl = []
        for i in range(0,len(c)-self.N_pushers,self.N_pushers):
            p = c[i:i+self.N_pushers]
            q = c[i+self.N_pushers:i+2*self.N_pushers]    
            kl_i = 0     
            for p_i, q_i in zip(p,q): 
                if p_i == 0 or q_i == 0:
                    kl_i += 0
                else:
                    kl_i += p_i*np.log(p_i/q_i)
            kl.append(kl_i)
        return kl
    
    def compute_hp(self,a, b, v):
        dp = -np.dot(a, v) - b
        return dp
    
    def poc_ch(self,theta):

        points_before_rotation = np.array([
            [0.1, 0, -0.1],  
            [-0.1, 0, -0.1],  
            [0.5, 0.1, 0.1], 
            [-0.5, 0.1, 0.1]  
            ])
        
        axis = [0,0,1]
        rotated_cone_points = np.vstack([rotate_cone(points_before_rotation, axis, theta), np.array([0, 0, 0])])
        
        #points_after_rotation = np.dot(points_before_rotation, R_y)
        #print("rotation: \n", points_after_rotation)
        points_after_rotation = np.vstack([points_before_rotation, np.array([0, 0, 0])])
        #plot_convex_hull(points_after_rotation)
        #plot_convex_hull(rotated_cone_points)
        convex_hull = ConvexHull(rotated_cone_points)

        return convex_hull
    def get_square_const(self,x, z, inequalities):
            inequalities.extend([-self.d/10+x])
            inequalities.extend([-self.d/10+z])
            inequalities.extend([-self.d/10-x])
            inequalities.extend([-self.d/10-z])
            return inequalities
     #center it where needed -> need to be generalized
    def sigmoid(self,x):
        x_ = x%.1
        return 1 / (1 + np.exp(-300 * (x + 0.02)))

    #TODO compute the constraints in a class object, to extend to multiple objects shape
    def get_t_constraints(self, x, z, constraints):

        c = self.sigmoid(x)
        epsilon = 0.003
        a = 0.015  - epsilon
        b = 0.05   - epsilon
  
        #using the soft constraint
        constraints.extend([c*(z+a)*(-z+a)+(1-c)*(z+b)*(-z+b)])
        constraints.extend([x + 0.05])
        constraints.extend([-x + 0.05])        


        return constraints


    
    def retrieve_optim_vels(self,v_x,v_z,v_w,pushers):
        vx = []
        vz = []
        vw = []
        for t in range(self.N_steps):
            best_pusher = -1
            idx = 0
            best_idx = 0
            for j in range(self.N_pushers):
                if pushers[t*self.N_pushers+j] > best_pusher:
                    best_pusher = pushers[t*self.N_pushers+j]
                    best_idx = idx
                idx+=1
            if t*self.N_pushers+best_idx < len(v_w):
                v_x_t = v_x[t*self.N_pushers+best_idx]
                v_z_t = v_z[t*self.N_pushers+best_idx]
                v_w_t = v_w[t*self.N_pushers+best_idx]
                vx.append(v_x_t)
                vz.append(v_z_t)
                vw.append(v_w_t) 
        return vx, vz, vw




    def eq_constraints(self, vars):
        
        raise NotImplementedError()

    def ineq_constraints(self, vars):
                
        raise NotImplementedError()
    
    def compute_positions(self, velocities_x, velocities_z, angular_velocities_w, T):
        
        raise NotImplementedError()

    def compute_positions_probabilities(self, velocities_x, velocities_z, velocities_w, T,c):
        
        raise NotImplementedError()

    def objective_function(self, vars):
        
        raise NotImplementedError()

    def callback_func(self, xk):
        
        raise NotImplementedError()

    def vels_ig(self, T_ig):
        
        raise NotImplementedError()

    def get_ig(self):
        
        raise NotImplementedError()

    def return_solution(self, result):
        
        raise NotImplementedError()
   
    def optimize_motion_cone(self):
        
        raise NotImplementedError()
