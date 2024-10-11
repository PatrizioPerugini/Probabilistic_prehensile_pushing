from rtree import index
import numpy as np
from model import Model
import time


class Node:
    def __init__(self, state,
                       parent=None, 
                       cost=0.,
                       dt = None,
                       from_rs=None,
                       reachable = None):
        self.parent:Node = parent
        self.children = set()
        self.cost = cost
        self.dt = dt
        self.state = state
        self.reachable = reachable if reachable is not None else []
        # it allows to store both information about who the fatehr is and 
        # from which reachables father's set we end up in this node
        self.from_rs = from_rs

    def cumulative_cost(self):
        cost = self.cost
        if self.parent is None:
            return 0
        return cost + self.parent.cumulative_cost()

    def add_child(self, child):
        if child in self.children:
            return False
        else:
            self.children.add(child)
            return True

    def __hash__(self) :#-> int:
        return hash(str(self.state))
    
    def __eq__(self, __o: object) :#-> bool:
        return self.__hash__() == __o.__hash__()

    def __repr__(self) :#-> str:
        return f"[{self.state}]"

class Planner:
    def __init__(self, model: Model, maximum_time, thr=1e-5, step_size = 0.1):
        #x_dim=3
        self.model = model
        self.x_dim = self.model.x_dim
        self.state_tree = StateTree(self.x_dim)
        self.n_nodes = 0
        self.thr = thr
        self.min_dist = np.inf
        self.id_to_node: dict[int, Node] = {}
        self.step_size = step_size
        self.maximum_time = maximum_time
        
    
    def nodes(self):
        return self.id_to_node.values()
    
    def get_plan(self, node):
        nodes = [node]
        while node.parent is not None:
            nodes = [node.parent] + nodes
            node  = node.parent
        return nodes

    def add_node(self,state, parent:Node = Node, cost = None, dt = None):
        raise NotImplementedError()
    
    def expand(self, x_rand: np.ndarray):#-> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def plan(self, max_nodes):
        initial_state = self.model.initial_state

        self.initial_node = self.add_node(initial_state)
        goal, distance = self.model.goal_check(self.model.initial_state)
        node_list=[]

        if(distance < self.min_dist):
            self.min_dist = distance
        if goal:
            plan=self.get_plan(self.initial_node)
            return True, plan, node_list
 
        
        it = 0
 
        cumulative_time = 0
        

        while self.n_nodes < max_nodes:
            
            start_time = time.time()
            x_rand = self.model.sample()
            node_next, node_near = self.expand(x_rand)
            stop_time = time.time()
            elapsed_time = stop_time - start_time
           # print("NODE NEXT IN PLANNER: \n", node_next)
            cumulative_time+= elapsed_time

            if cumulative_time > self.maximum_time:
                print("Maximum time reached, failure")
                return False, None, True #we use the true here to indicate that the maximum time has been surpassed
            if node_next is None:
                #dropped+=1
                continue
            node_list.append(node_next.state)
            ########################## USE IT LATER ON
            print("new node is: ", node_next.state, " min dist: ", self.min_dist, " pusher: ", node_next.from_rs)
            goal, plan = self.goal_check(node_next)

            if it%20 == 0:
                print(f"n_nodes: {self.n_nodes}, dist: {self.min_dist}, new_node: {node_next}", end='\r')
            it+=1
            if goal:
                print("reached the goal with a distance of: ", self.min_dist)
                return goal, plan, node_list
            
        return False, None, None
    
    def goal_check(self, node):
        state = node.state
        goal,distance = self.model.goal_check(state)
        if distance < self.min_dist:
            self.min_dist = distance
        if goal:
            plan = self.get_plan(node)
            return True, plan
        return False, None

class StateTree():
    def __init__(self, dim) :
        self.dim = dim
        self.state_tree_p = index.Property()
        self.state_tree_p.dimension = dim
        self.state_idx = index.Index(properties = self.state_tree_p, 
                                    interleaved = True)
    
    def insert(self, state):
        state_id = self._id(state)
        self.state_idx.insert(state_id,state)
        return state_id
    
    def nearest(self, state):
        nearest_id = list(self.state_idx.nearest(state, num_results=1))[0]

        return nearest_id

    def _id(self,x):
        return hash(str(x))