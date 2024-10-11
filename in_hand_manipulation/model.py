import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from utils.utils import plot_convex_hull, rotate_cone
import sympy as sp

class Model:
    def __init__(self, shape, x_dim =3, m = 1, g=9.81, 
                initial_state=np.array([-0.01,-0.01,0]), 
                goal_state=np.array([0.2,0.3,-0.0]), eps_goal = 1e-3, dt = 1 ) -> None:
        
        #configuration space
        self.x_dim = x_dim
        
        #object used
        self.shape = shape
        self.m = m
        self.g = g
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.eps_goal = eps_goal
        self.dt = dt
        #parameters of the robot and the environment
        self.r = 1/100  
        self.rc_q = (0.6 * self.r) ** 2
        self.m = 500/1000
        self.mu_s = 0.9
        self.N=50
        self.mu = 0.9
        self.theta_mu =math.atan(self.mu)
        self.c = math.cos(self.theta_mu)
        self.s = math.sin(self.theta_mu)
        self.w = self.shape.width
        self.d = self.shape.width/2
        #self.d = 5/100  #cm
        #self.w = 10/100#cm
        self.B = np.diag([1,1,1/self.rc_q])


        #ADD THE DIFFERENT WALL ORIENTATIONS IN THE ENVIRONMENT
        self.wall_orientations = [0,-np.pi/2, np.pi/2]


    
    def compute_rigid_transform(self,x):
        #input: state (x,z,theta)
        #output: transform from world to object frame
        
        c = math.cos(x[2])
        s = math.sin(x[2])
        tx = x[0]
        tz = x[1]
        T = np.array([[c, -s, tx],[s, c, tz], [0, 0, 1]])
        return T

  
    #NB-> it should take into account all the different pushers that are available... 
    #  compute the polyhedral motion cones for a given object configuration in 
    # the grasp for all possible external pushers
    
    #x:np.ndarray->state
    def get_motionCones(self, x, theta_rad): 
        fx_h, fz_h, my_h, k = sp.symbols("fx_h fz_h my_h k")
        num_sol = 4
        J_s = self.compute_rigid_transform(x)

        f_r = [np.cos(np.pi/2-math.atan(self.mu)),np.sin(np.pi/2-math.atan(self.mu))]
        f_l = [-np.cos(np.pi/2-math.atan(self.mu)),np.sin(np.pi/2-math.atan(self.mu))]

        
        J_p1 = np.array([[1, 0], 
                     [0, 1], 
                     [-self.w/2, self.d]])
        J_p2 = np.array([[1, 0], [0, 1], [-self.w/2, -self.d ]])

        wp1_r = np.dot(J_p1, f_r)
        wp1_l = np.dot(J_p1, f_l)
        wp2_r = np.dot(J_p2, f_r)
        wp2_l = np.dot(J_p2, f_l)

        norm = self.N*self.mu,
        wrench_cone = np.vstack([wp1_r, wp1_l, wp2_r, wp2_l])
        sol = []
        for edge_wrench in wrench_cone:
            w_pusher_hat=sp.Matrix([edge_wrench[0]/norm, edge_wrench[1]/norm, edge_wrench[2]/norm])
            fx_h_vec = sp.Matrix([fx_h, fz_h, my_h])#w_s_hat                                                           # sp.Matrix([0, 0, -9.8])
                          #J_s
            stab_push_eq = J_s.T * fx_h_vec - k/(-self.mu_s*self.N) * w_pusher_hat - self.m/(-self.mu_s*self.N) * sp.Matrix([0, 9.8, 0 ])
   
            eq_ellips = sp.Eq(fx_h**2/1 + fz_h**2/1 + my_h**2/self.rc_q, 1)

            solution = sp.solve([eq_ellips, stab_push_eq], [fx_h, fz_h, my_h, k])
            if(solution[0][num_sol-1]> 0):
                sol.append(solution[0])
            else:
                sol.append(solution[1])

        solutions = np.array(sol)

        ws_h = []
        for i in range(len(sol)):
            ws_h.append(solutions[i][0:num_sol-1])
        v_obj = []
        for ws in ws_h:
            ws = np.reshape(ws,(3,1))
            v_i = np.linalg.inv(J_s)@self.B@(ws)
            #v_i = J_s@self.B@(ws)
            v_obj.append(v_i)

        V_obj = np.reshape(np.array(v_obj,dtype = float), (num_sol,3,))  
        
        V_origin =np.array([0,0,0])
        check_hull_vel = np.vstack([V_obj[0][:],V_obj[1][:],V_obj[2],V_obj[3]])

        axis = [0,0,1]
        rotated_cone_points = np.vstack([rotate_cone(check_hull_vel, axis, theta_rad), V_origin])
        hull = ConvexHull(rotated_cone_points)


        return hull

    def goal_check(self, x):
        min_dist = np.inf
        x_ = x.copy()
        goal = False

        dist_xz = np.linalg.norm(x_[0:2]-self.goal_state[0:2])
        dist_y = np.linalg.norm(x_[2]-self.goal_state[2])
        dist = dist_xz + dist_y*0.01
        #dist = 0.1*dist_y + 0.9*dist_xz
        if dist<min_dist:
            min_dist = dist

        if min_dist < self.eps_goal:
            goal = True
        
        return goal, min_dist
        
    

    def sample(self):
        goal_bias = np.random.rand(1)
 
        if goal_bias < 0.1: return self.goal_state
  
        rnd = (np.random.rand(3)-0.5)*2# range between -1 and 1

        rnd[2]= (np.random.rand(1) - 0.5) * 0.7 * np.pi/4
   
        return rnd