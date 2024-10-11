import numpy as np
from matplotlib.patches import Rectangle

#define any desired shape: 
#The constraints for the optimizer are defined within each object accordingly

class Shape:
    def __init__(self, width = 0.1, height = 0.1):
        self.width = width
        self.height = height
        self.name = "square"

    def sample(self):
        raise NotImplementedError()
    
    def get_meshes(self):
        raise NotImplementedError()

    def get_constraints(self, x, z, constraints):
        raise NotImplementedError()

class TShape(Shape):
    def __init__(self, width = 0.1, height = 0.1 , width_2 = 0.03, switch_threshold = 0.02):
        super().__init__(width, height)
        self.width_2 = width_2 
        self.switch_threshold = switch_threshold
        self.name ="T"
    def sample(self):

        eps = 0.0005
        min_val = -self.width/2
        max_val =  self.width/2
        x_start =  np.random.rand() * (max_val - min_val) + min_val
        z_start = np.random.rand()* (max_val - min_val) + min_val

        x_goal =  np.random.rand() * (max_val - min_val) + min_val
        z_goal = np.random.rand()* (max_val - min_val) + min_val
        z_r_max =  self.width_2/2
        z_r_min = -self.width_2/2
        if x_goal > -self.switch_threshold :#right side of the t
            z_goal = np.random.rand()* (z_r_max - z_r_min) + z_r_min
        if x_start > -self.switch_threshold :#right side of the t
            z_start = np.random.rand()* (z_r_max - z_r_min) + z_r_min

        theta_start = (np.random.rand() - max_val)/3
        theta_goal =  (np.random.rand() - max_val)/3
        
        start_position = np.array([x_start, z_start, theta_start])
        goal_position = np.array([x_goal, z_goal, theta_goal])

        return start_position, goal_position


    def sigmoid(self,x):
        x_ = x%.1
        return 1 / (1 + np.exp(-300 * (x + self.switch_threshold)))
    

    def get_constraints(self, x, z, constraints):

        c = self.sigmoid(x)
        epsilon = 0.003
        a = self.width_2/2  - epsilon
        b = self.width/2   - epsilon
  
        #using the soft constraint
        constraints.extend([c*(z+a)*(-z+a)+(1-c)*(z+b)*(-z+b)])
        constraints.extend([x + self.width/2])
        constraints.extend([-x + self.width/2])        
        epsilon = 0.003
 

        return constraints
    
    def get_meshes(self):
        meshes = []
        vertical_part = Rectangle((-self.switch_threshold, -self.width_2/2), self.width - self.width_2, self.width_2, linewidth=3, edgecolor='#001b2e', facecolor='none')
        horizontal_part = Rectangle((-self.width/2, -self.width/2), self.width_2, self.width, linewidth=3, edgecolor='#001b2e', facecolor='none')
        meshes.append(vertical_part)
        meshes.append(horizontal_part)
        return meshes

class SShape(Shape):
    def __init__(self, width=0.1, height=0.1):
        super().__init__(width, height)

        self.width = width
        self.height = height

    def sample(self):
        eps = 0.0005
        min_val = -self.width/2
        max_val =  self.width/2
        x_start =  np.random.rand() * (max_val - min_val) + min_val
        z_start = np.random.rand()* (max_val - min_val) + min_val

        x_goal =  np.random.rand() * (max_val - min_val) + min_val
        z_goal = np.random.rand()* (max_val - min_val) + min_val

        theta_start = (np.random.rand() - max_val)/3
        theta_goal =  (np.random.rand() - max_val)/3
        
        start_position = np.array([x_start, z_start, theta_start])
        goal_position = np.array([x_goal, z_goal, theta_goal])

        return start_position, goal_position
    
    def get_meshes(self):
        meshes = []
        r = Rectangle((-self.width/2, -self.width/2), self.width, self.width, linewidth=3, edgecolor='#001b2e', facecolor='none')
        meshes.append(r)
        return meshes

    def get_constraints(self, x, z, constraints):
        
        constraints.extend([x+self.width/2])
        constraints.extend([-x+self.width/2])
        constraints.extend([z+self.height/2 ])
        constraints.extend([-z+self.height/2])
        return constraints