import sympy as sp
import numpy as np
import math
import dill
import os
import gurobipy as gp
from gurobipy import GRB

dill.settings['recurse'] = True

class MotionConeG:
    
    def __init__(self):
        pass

    def retrieve_solution(self, c, s, t_x, t_z, a1, a2, a3, b, r_c):

        model = gp.Model("OptimizationModel")
        F_x   = model.addVar(name="F_x")
        F_z   = model.addVar(name="F_z")
        M_y   = model.addVar(name="M_y")
        K = model.addVar(name="K")
        model.addConstr(c * F_x + s * F_z - K * a1 == 0, name="eq1")
        model.addConstr(-s * F_x + c * F_z - K * a2 - b == 0, name="eq2")
        model.addConstr(F_x**2 + F_z**2 + (M_y**2 / r_c) == 1, name="eq4")

        model.addConstr(t_x * F_x + t_z * F_z + M_y - K * a3 == 0, name="eq3")

        model.setObjective(0, gp.GRB.MINIMIZE)

        model.optimize()

        return F_x, F_z, M_y, K

class MotionCone:

    def __init__(self) -> None:
       
        # Convert the solution to a Python function
        self.F_x_func_0 = None
        self.F_z_func_0 = None
        self.M_y_func_0 = None
        self.K_func_0   = None
       

        self.F_x_func_1 = None
        self.F_z_func_1 = None
        self.M_y_func_1 = None
        self.K_func_1   = None


    def retrieve_solution(self,c_val,s_val,t_x_val,t_z_val,a1_val,a2_val,a3_val,b_val,r_c_val):
        F_x_val_0 = self.F_x_func_0(c_val, s_val, t_x_val, t_z_val, a1_val, a2_val, a3_val, b_val, r_c_val)
        F_z_val_0 = self.F_z_func_0(c_val, s_val, t_x_val, t_z_val, a1_val, a2_val, a3_val, b_val, r_c_val)
        M_y_val_0 = self.M_y_func_0(c_val, s_val, t_x_val, t_z_val, a1_val, a2_val, a3_val, b_val, r_c_val)
        K_val_0 = self.K_func_0(c_val, s_val, t_x_val, t_z_val, a1_val, a2_val, a3_val, b_val, r_c_val)
        if K_val_0 > 0:
            return F_x_val_0, F_z_val_0, M_y_val_0, K_val_0
        else:
            F_x_val_1 = self.F_x_func_1(c_val, s_val, t_x_val, t_z_val, a1_val, a2_val, a3_val, b_val, r_c_val)
            F_z_val_1 = self.F_z_func_1(c_val, s_val, t_x_val, t_z_val, a1_val, a2_val, a3_val, b_val, r_c_val)
            M_y_val_1 = self.M_y_func_1(c_val, s_val, t_x_val, t_z_val, a1_val, a2_val, a3_val, b_val, r_c_val)
            K_val_1 = self.K_func_1(c_val, s_val, t_x_val, t_z_val, a1_val, a2_val, a3_val, b_val, r_c_val)
            return F_x_val_1, F_z_val_1, M_y_val_1, K_val_1

    def __get_lambdified_functions(self):
        F_x, F_z, M_y, K, c, s, t_x, t_z, a1, a2, a3, b, r_c = sp.symbols('F_x F_z M_y K c s t_x t_z a1 a2 a3 b r_c')
        # Define your equations here
        eq1 = sp.Eq(c * F_x + s * F_z - K * a1, 0)
        eq2 = sp.Eq(-s * F_x + c * F_z - K * a2 - b, 0)
        eq3 = sp.Eq(t_x * F_x + t_z * F_z + M_y - K * a3, 0)
        eq4 = sp.Eq(F_x**2 + F_z**2 + (M_y**2 / r_c), 1)

        # Solve the equations symbolically
        solutions = sp.solve((eq1, eq2, eq3, eq4), (F_x, F_z, M_y, K))

        # Define lambdified functions
        F_x_func_0 = sp.lambdify((c, s, t_x, t_z, a1, a2, a3, b, r_c), solutions[0][0])
        F_z_func_0 = sp.lambdify((c, s, t_x, t_z, a1, a2, a3, b, r_c), solutions[0][1])
        M_y_func_0 = sp.lambdify((c, s, t_x, t_z, a1, a2, a3, b, r_c), solutions[0][2])
        K_func_0 = sp.lambdify((c, s, t_x, t_z, a1, a2, a3, b, r_c), solutions[0][3])

        F_x_func_1 = sp.lambdify((c, s, t_x, t_z, a1, a2, a3, b, r_c), solutions[1][0])
        F_z_func_1 = sp.lambdify((c, s, t_x, t_z, a1, a2, a3, b, r_c), solutions[1][1])
        M_y_func_1 = sp.lambdify((c, s, t_x, t_z, a1, a2, a3, b, r_c), solutions[1][2])
        K_func_1 = sp.lambdify((c, s, t_x, t_z, a1, a2, a3, b, r_c), solutions[1][3])

        return (F_x_func_0, F_z_func_0, M_y_func_0, K_func_0,
                F_x_func_1, F_z_func_1, M_y_func_1, K_func_1)

    def __save_functions_to_pickle(self, path ):
              # Define symbols
        if not os.path.exists(path):   
            with open(path, 'wb') as f:
                functions = self.__get_lambdified_functions()
                dill.dump(functions, f)
            print(f"Functions saved to {path}")
        else:
            print(f"The file {path} already exists.")


    def get_functions(self, path = "lambdified_functions.pkl" ):
        if not os.path.exists(path):
            self.__save_functions_to_pickle(path)
        
        with open(path, 'rb') as f:
            (self.F_x_func_0, self.F_z_func_0, self.M_y_func_0, self.K_func_0,
             self.F_x_func_1, self.F_z_func_1, self.M_y_func_1, self.K_func_1) = dill.load(f)
        



