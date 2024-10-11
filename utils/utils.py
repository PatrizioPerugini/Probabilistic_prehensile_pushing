from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, LinearConstraint
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
import os
from copy import copy
import pickle


def random_points_test(x_min_start = 0, x_max_start = 0.03, 
                     y_min_start = -0.005, y_max_start = 0.005, 
                     theta_min_start = -0.2, theta_max_start = 0.2, 
                     x_min_goal = -0.045, x_max_goal = -0.025, 
                     y_min_goal = -0.04, y_max_goal = 0.02, 
                     num_points = 50):
    seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    # Generate random points within the square range
    start = np.random.uniform(low=(x_min_start, y_min_start,theta_min_start), high=(x_max_start, y_max_start, theta_max_start), size=(num_points, 3))
    goal = np.random.uniform(low=(x_min_goal, y_min_goal, theta_min_start), high=(x_max_goal, y_max_goal, theta_max_start), size=(num_points, 3))
    return start, goal
    
def is_grasped(x_pos,y_pos, shape):
    
    if shape.name == "square":
        if x_pos > -shape.width/2 and x_pos < shape.width/2 and y_pos > -shape.width/2 and y_pos < shape.width/2:
            return True
        else:
            return False
    else: #shape it T-SHAPE
        if x_pos > -shape.height/2 and x_pos < -shape.switch_threshold:
            if y_pos > -shape.height/2 and y_pos < shape.height/2:
                return True
            else:
                return False
        elif x_pos > -shape.switch_threshold and x_pos < shape.height/2:
            if y_pos > -shape.switch_threshold+0.005 and y_pos < shape.switch_threshold-0.005:
                return True
            else:
                return False
        else:
            return False




#Add the node only if it brings me closer
def transition_test(q_parent, q_sample, q_goal, treshold = 0.0):

    pre_distance = np.linalg.norm(q_parent-q_goal)
    post_distance = np.linalg.norm(q_sample-q_goal)
    
    return post_distance+treshold<pre_distance

def rotation_matrix(axis, theta):
 
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta/2 )
    b, c, d = -axis * np.sin(theta/2 )
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])


def rotate_cone(points, axis, theta):
   
    R = rotation_matrix(axis, theta)
    return np.dot(points, R.T)

def plot_convex_hull(points , hull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Points')
    
    # Plot the convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the simplex
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')
        
    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Show the plot
    plt.show()

def plot_hull_and_points(hull, point, projected_point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot convex hull
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], hull.points[simplex, 2], 'k-')

    # Plot the point
    ax.scatter(point[0], point[1], point[2], color='red', label='original point')
    ax.scatter(projected_point[0], projected_point[1], projected_point[2], color='green', s = 7, label='projected point')

    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


#If the sampled state is not in the motion cone, 
#project the twist into the closest one

def objective(x, original_point):
    return np.sum((x - original_point)**2)

def find_closest_projection(hull, original_point):
    hull_equations = hull.equations[:, :3]
    hull_biases = -hull.equations[:, 3] 
    # Set up linear inequality constraints Ax <= b
    A = hull_equations
    b = hull_biases

    bounds = [(None, None)] * 3 

    linear_constraint = LinearConstraint(A, -np.inf, b, keep_feasible=False)

    result = minimize(objective, original_point, args = (original_point, ), constraints=[linear_constraint], bounds=bounds)

    projected_point = result.x

    distance = np.linalg.norm(original_point - projected_point)

    return projected_point, distance


def project_onto_convex_hull(hull, point):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot convex hull
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], hull.points[simplex, 2], 'k-')
        
    # Plot the point
    ax.scatter(point[0], point[1], point[2], color='red', label='Point to Test')

    closest_projection, dist = find_closest_projection(hull, point)
    ax.plot([point[0], closest_projection[0]], [point[1], closest_projection[1]], [point[2], closest_projection[2]], "c<:")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


################ PROJECTION STUFF

def plot_hull_and_point(hull, point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    proj_point = project_onto_convex_hull(hull, point)

    # Plot convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the loop
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], hull.points[simplex, 2], 'k-')

    # Plot the point
    ax.scatter(point[0], point[1], point[2], color='red', label='Point to Test')
    ax.scatter(*proj_point, color='green', label='Point to proj')


    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

def test_hull(): 
    points = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    hull = ConvexHull(points)

    # Test points
    test_point_inside = np.array([0.5, 0.5])
    test_point_outside = np.array([2, 2])

    # Check if points are inside the convex hull
    result_inside = point_in_hull(test_point_inside, hull)
    result_outside = point_in_hull(test_point_outside, hull)

    print(f"Is {test_point_inside} inside the convex hull? {result_inside}")
    print(f"Is {test_point_outside} inside the convex hull? {result_outside}")

def plot_moving_square_with_rotation(path, rotation_angles,pushers, gt = [-0.1,0.2,-np.pi/8],circular_object_radius=0.2, dir="./"):
    gt_x , gt_z, gt_theta = gt
    print(pushers)
    print(gt)
    for i in range(len(path)):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        wall = pushers[i]
        # Plot the circular object (end effector) at the fixed position (0, 0)
        circular_object_theta = np.linspace(0, 2 * np.pi, 100)
        circular_object_x = circular_object_radius * np.cos(circular_object_theta)
        circular_object_y = circular_object_radius * np.sin(circular_object_theta)
        ax.plot(circular_object_x, circular_object_y, color='red', label='End Effector (Fixed)')

        state_x, state_z = path[i]
        rotation_angle = rotation_angles[i] 

        # Rotate the square object and the wall around the y-axis
        rotation_matrix_y = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                                      [0, 1, 0],
                                      [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]])

        rotation_matrix_y_GT = np.array([[np.cos(gt_theta), 0, np.sin(gt_theta)],
                                      [0, 1, 0],
                                      [-np.sin(gt_theta), 0, np.cos(gt_theta)]])

        square_vertices = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5],[-0.5, 0.5]])
        rotated_square = np.dot(np.column_stack([square_vertices, np.zeros((5, 1))]),
                                rotation_matrix_y.T)[:, :2] + np.array([state_x, state_z])

        rotated_square_GT = np.dot(np.column_stack([square_vertices, np.zeros((5, 1))]),
                                rotation_matrix_y_GT.T)[:, :2] + np.array([gt_x, gt_z])

        # Plot the rotated square object
        ax.plot(rotated_square[:, 0], rotated_square[:, 1], color='blue', label='Moving Tilted Object')

        ax.plot(rotated_square_GT[:, 0], rotated_square_GT[:, 1], color='green', label='GT pose')

        ### Plot the side of the square initially in contact with the wall



           ##if pusher below
        rotation_pusher = 0#-np.pi/2
      
        rotation_matrix_y_pusher = np.array([[np.cos(rotation_pusher), -np.sin(rotation_pusher), 0],
                                     [np.sin(rotation_pusher), np.cos(rotation_pusher), 0],
                                     [0, 0, 1]])

        rot_below = rotation_matrix_y_pusher@rotation_matrix_y
        
        wall_height = 1  # Adjust this value to set the height of the wall
        # if orizontal wall
        if wall == 0 :
            initial_wall_start = np.array([ state_x - 0.5 , state_z-0.5, 0])
            initial_wall_end = np.array([ state_x + wall_height - 0.5 , state_z-0.5, 0])      
            rotated_wall_start = np.dot(initial_wall_start, rotation_matrix_y.T)[:2]
            rotated_wall_end = np.dot(initial_wall_end, rotation_matrix_y.T)[:2]

        # Plot the rotated wall as a vertical line
        #elif vertical wall:
        elif wall == 2:
            rotated_wall_start = np.dot([0.5 + state_x, state_z - wall_height / 2, 0], rotation_matrix_y.T)[:2]
            rotated_wall_end = np.dot([0.5 + state_x, state_z + wall_height / 2, 0], rotation_matrix_y.T)[:2]
            rotated_wall_start = np.dot([0.5 + state_x, state_z - wall_height / 2, 0], rot_below.T)[:2]
            rotated_wall_end = np.dot([0.5 + state_x, state_z + wall_height / 2, 0], rot_below.T)[:2]
        else:
            rotated_wall_start = np.dot([-0.5 + state_x, state_z - wall_height / 2, 0], rotation_matrix_y.T)[:2]
            rotated_wall_end = np.dot([-0.5 + state_x, state_z + wall_height / 2, 0], rotation_matrix_y.T)[:2]
            #rotated_wall_start = np.dot([0.5 + state_x, state_z - wall_height / 2, 0], rot_below.T)[:2]
            #rotated_wall_end = np.dot([0.5 + state_x, state_z + wall_height / 2, 0], rot_below.T)[:2]


        ##

        ax.plot([rotated_wall_start[0], rotated_wall_end[0]], [rotated_wall_start[1], rotated_wall_end[1]],
                color='red', linestyle='-', linewidth=8, label='Rotated Wall')
                

        rotated_wall_start_gt = np.dot([ gt_x - 0.5 , gt_z-0.5, 0], rotation_matrix_y_GT.T)[:2]
        rotated_wall_end_gt = np.dot([ gt_x + wall_height - 0.5 , gt_z-0.5, 0], rotation_matrix_y_GT.T)[:2] 

        #if vertical wall
        #rotated_wall_start_gt = np.dot([0.5 + gt_x, gt_z - wall_height / 2, 0], rotation_matrix_y_GT.T)[:2]
        #rotated_wall_end_gt = np.dot([0.5 + gt_x, gt_z + wall_height / 2, 0], rotation_matrix_y_GT.T)[:2]
        
        ax.plot([rotated_wall_start_gt[0], rotated_wall_end_gt[0]], [rotated_wall_start_gt[1], rotated_wall_end_gt[1]],
                color='#FF9999', linestyle='-', linewidth=8, label='Rotated Wall_ gt')

    
        # Set labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        ax.legend()

        # Save the figure
        fig.savefig(f"{dir}/frame" + str(i) + ".png")

        # Show the plot
        plt.show()

        plt.close()


def edit_video(path,N,dt, speed=1.0):

    img_array = []
    img_path = path + "/frame"
    for n in range(N):
        filename = img_path + str(n)+ '.png'
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    fps = int(1/dt)*speed
    out = cv2.VideoWriter(path+'/video.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return out

def plan_check():
    plan = [
        [0, 0, 0],
        [-0.04396994, 0.05483787, -0.09919873],
        [-0.05593095, 0.09455506, -0.0018339331],
        [-0.01, 0.20194937, 0]#-0.35798662]
    ]

    pushers = [0, 1, 2, None]

    path = [point[:2] for point in plan]
    angles = [point[2] for point in plan]

#########################



def t2v(A):
    v = np.zeros((3, 1))
    v[0:2, 0] = A[0:2, 2]
    v[2, 0] = np.arctan2(A[1, 0], A[0, 0])
    return v

def v2t(v):
    c = np.cos(v[2])
    s = np.sin(v[2])
    A = np.array([[c, -s, v[0]],
                  [s, c, v[1]],
                  [0, 0, 1]])

    return A

def box_plus(p_n_x, p_n_z, p_n_w, dx, dz, dw):
    transf_incr = v2t([ dx, dz, dw])
    transf = v2t([ p_n_x, p_n_z, p_n_w])
    update = transf_incr@transf
    res = t2v(update)
    res = np.array([p_n_x, p_n_z, p_n_w]) + np.array([dx, dz, dw]) 
    return res 


def box_minus(self, T1, T2):

    inv_T1 = np.linalg.inv(T1)
    delta_T = inv_T1 @ T2
    return self.t2v(delta_T)

def check_p_in_cone(self,current_state,current_twist):
    motion_cones, _ = self.get_motionCones(current_state, theta_rad=0, vel_i= current_twist)
    for equation in motion_cones.equations:
        a = equation[:-1]  
        b = equation[-1]  
        is_inside = (self.cumput_hp(a, b, current_twist) >=0)
        if not is_inside:
            return False
    return True

def reset_axes(ax):
    ax.clear() 


def plot_solution(optimized_state, goal_position, meshes ,it_num=0, N = 0, pushers = 0, save_path = "images/fixed_t"):

    arr_x = [coord[0] for coord in optimized_state]
    arr_y = [coord[1] for coord in optimized_state]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(arr_x, arr_y, marker='o', label='Gripper Position', linewidth = 3, color = "tab:blue")  
    ax.scatter(goal_position[0], goal_position[1], color='tab:orange', label='Goal Position', zorder=10)
    
    # Add T shape
    for c in meshes:
       new_c=copy(c)
       ax.add_patch(new_c)
    
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    #sax.set_title(f'c: {pushers}')
    ax.set_aspect('equal')
    ax.legend()

    # Create directory if not existing
    os.makedirs(save_path, exist_ok=True)
    name = "N_" + str(N) + "_it_" + str(it_num)+ ".png"
    # Save the figure
    save_file = os.path.join(save_path, name)
    plt.savefig(save_file)
    print("figure saved")
    plt.close(fig)
    #plt.close()
    #fig.close()

def rotation_matrix(axis, theta):
    """
    Create a rotation matrix for rotating `theta` radians around the specified `axis`.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta/2 )
    b, c, d = -axis * np.sin(theta/2 )
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])


def plot_cone(ax, points, color):
    hull = ConvexHull(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 'o', color=color)

    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # To create a closed loop
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], color+'-')

    vertices = [points[simplex] for simplex in hull.simplices]
    hull_faces = Poly3DCollection(vertices, alpha=0.3, facecolor=color, linewidths=1, edgecolors='r')
    ax.add_collection3d(hull_faces)

def main_plot_cones():
    # Define the initial points (vertices of the convex hull), including the origin
    initial_cone = np.array([
        [0, 0, 0],  # Origin
        [1, 0, -1],  # Positive x-direction
        [-1, 0, -1],  # Negative x-direction
        [0.5, -1, 1],  # Negative y-direction
        [-0.5, -1, 1]  # Negative y-direction
    ])

    # Define the wall orientations
    # Example: parallel to the ground, pushing from above
    #wall_orientation_top = [1, 0, 0]  # Rotate around x-axis
    wall_orientation_right = [0, 0, 1]  # Rotate around x-axis
    ## Example: pushing from the left side
    #wall_orientation_left = [0, 1, 0]  # Rotate around y-axis
    wall_orientation_left = [0, 0, 1]
    # Rotate the cone for the top wall orientation
    theta_right = -np.pi / 2  # 90 degrees
    rotated_cone_right = rotate_cone(initial_cone, wall_orientation_right, theta_right)

    # Rotate the cone for the left wall orientation
    theta_left = np.pi / 2  # 90 degrees
    rotated_cone_left = rotate_cone(initial_cone, wall_orientation_left, theta_left)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the initial and rotated cones
    plot_cone(ax, initial_cone, 'c')
    plot_cone(ax, rotated_cone_right, 'm')
    plot_cone(ax, rotated_cone_left, 'y')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def plot_convex_hull_3d(points, convex_hull_inequalities, point_to_check=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Points')

    # Plot convex hull faces
    for normal, d in convex_hull_inequalities:
        if normal[2] != 0:  # Check if normal[2] is not zero to avoid division by zero
            xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
            zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            ax.plot_surface(xx, yy, zz, alpha=0.2)

    # Plot convex hull edges
    for i in range(len(points)):
        for j in range(i, len(points)):
            ax.plot([points[i][0]], [points[i][1]], [points[i][2]], 'ro')  # Plot points
            #j = (i + 1) % len(points)
            ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], [points[i][2], points[j][2]], 'g-')  # Plot edges

    # Plot point to check if provided
    if point_to_check is not None:
        ax.scatter(point_to_check[0], point_to_check[1], point_to_check[2], c='r', marker='*', s=100, label='Point to Check')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Convex Hull')
    ax.legend()
    plt.show()

def orientation_3d(p, q, r):
    """
    Determine the orientation of triplet (p, q, r) in 3D space.
    Returns:
    - 0 if colinear
    - 1 if clockwise
    - 2 if counterclockwise
    """
    #val = np.cross_product(p.subtract(r), q.subtract(r))
    #print("val: ", val)
    val = (q[1] - p[1]) * (r[2] - q[2]) - (q[2] - p[2]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def jarvis_march_3d(points):
    """
    Compute the convex hull in 3D space using the Gift Wrapping (Jarvis March) algorithm.
    Returns:
    - convex_hull_inequalities: List of inequalities representing the half-spaces of the convex hull.
    """
    n = len(points)
    if n < 4:
        raise ValueError("Convex hull in 3D space requires at least 4 points")

    if isinstance(points, np.ndarray):
        points = points.tolist()
    hull_inequalities = []

    # Find the leftmost point
    leftmost = min(points, key=lambda x: x[0])

    hull = []
    p = points.index(leftmost)
    q = 0

    while True:
        hull.append(points[p])

        q = (p + 1) % n
        for i in range(n):
            if orientation_3d(points[p], points[i], points[q]) == 2:
                q = i

        p = q

        if p == 0:
            break

    # Compute plane equations for each face of the convex hull
    for i in range(len(hull) - 1):
        v1 = hull[i]
        v2 = hull[i + 1]
        normal = (v2[1] - v1[1], v1[0] - v2[0], 0)  # Cross product with Z-axis to get normal vector
        d = -(normal[0] * v1[0] + normal[1] * v1[1] + normal[2] * v1[2])
        hull_inequalities.append((normal, d))

    # Handle the last face
    v1 = hull[-1]
    v2 = hull[0]
    normal = (v2[1] - v1[1], v1[0] - v2[0], 0)
    d = -(normal[0] * v1[0] + normal[1] * v1[1] + normal[2] * v1[2])
    hull_inequalities.append((normal, d))

    return hull_inequalities

def write_results_to_file(results, filename):
    with open(filename, 'wb') as f:  # Open file in binary write mode
        pickle.dump(results, f)  # Serialize dictionary and write to file
        
def draw_convex_hull_3d(points, hull, point = None) -> None:
    for i, p in enumerate(points):
        print(f'p: {p} - {i}')
    i = 0
    col = ['r','b','g','r','d']
  
    for simplex in hull.simplices:
        #axs[i].scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='o')
        #col_i = col[i%len(col)]
        #axs[i].plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], col_i+'-',)
        i+=1
    #plt.show()
    i = 0
    fig = plt.figure('Convex hull computation')
    
    ax = fig.add_subplot(111, projection='3d')
   #     draw_grid_3d()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='o')
    if point is not None:
        ax.scatter(*point, c='k', marker='x')
    for simplex in hull.simplices:
        col_i = col[i%len(col)]
       
        triangle = Poly3DCollection([points[simplex]], facecolors='cyan', edgecolors='r', alpha=0.5)
        ax.add_collection3d(triangle)

        i+=1
    plt.show()




def is_inside_convex_hull(point, convex_hull_inequalities):
    """
    Check if a point is inside the convex hull defined by the inequalities.
    Returns:
    - True if the point is inside or on the boundary of the convex hull.
    - False otherwise.
    """
    for normal, d in convex_hull_inequalities:
        if np.dot(normal, point) + d > 0:
            return False
    return True

def plot_path(goal, plan, node_list, m):
    plot_list = np.array(plan)
    if plan is not None:
        print("The PLAN is: \n ", plan)
        r = m.r
        path = [(node.state[0], node.state[1]) for node in plan]
        angles = [node.state[2] for node in plan]
        pushers = [node.from_rs for node in plan if node.from_rs is not None]
        print("pushers used: ", pushers)

        pushers.append(0)
        gt = m.goal_state#.tolist()

        x_values, y_values = zip(*path)

        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, marker='o')
        plt.title('Path of Coordinates')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()
        print('n states: ',len(node_list))
    else:
        print("Unable to find path")
    
if __name__ == "__main__":
    # Example usage
    try_ = False
    if try_:
        num_points = 10
        min_coord, max_coord = -1, 1
        points_3d = [(0, 0, 3),(-1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0) ]
        convex_hull_inequalities = jarvis_march_3d(points_3d)
        print(convex_hull_inequalities)
        random_points = np.random.uniform(min_coord, max_coord, size=(num_points, 3))
        for point_to_check in random_points:
        #point_to_check = np.array([-0.5, 0, 0.7])  # Point to check
            print("The point is: ", point_to_check)
            inside = is_inside_convex_hull(point_to_check, convex_hull_inequalities)
            if inside:
                print("Point is inside or on the boundary of the convex hull.")
            else:
                print("Point is outside the convex hull.")

            plot_convex_hull_3d(points_3d, convex_hull_inequalities, point_to_check=point_to_check)
