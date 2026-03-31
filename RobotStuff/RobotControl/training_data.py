import numpy as np
import torch
from forward_kinematics import Robot_math
from path_planning import PathPlanner


"""
The neural network takes inputs of 
q_t_-_1, q_t, G_d, G_6D_R
and outputs q_t_+_1


Training on a mix of valid random MoveL trajectories and random points in the workspace. 



Random MoveL, which generates a trajectory of a straight line between p0 and p1.
p0 and p1 are generated from a random q_vect_0 and q_vect_1 passed throught forward kinematics
Then we generate a straight smooth position and orientation trajectory of 20 points between p0 and p1,
and train the network to predict 
the sequence of q_vect that would achieve that trajectory sequentially



If moving to a random point in the workspace, prev_q = curr_q (no initial velocity)
"""



class make_robot_data:
    def __init__(self, my_robot, my_path_planner):
        self.rob = my_robot
        self.PathPlan =  my_path_planner
        pass

   
    def MoveL_traj(self, f, t, p0=None, p1=None):
        '''
        Interpolate p0 to p1 over timeframe t in f frames.
        p0 is flattened start xyz and start SO3_R
        p1 is flattened goal xyz and goal SO3_R

        if p0 is None, make a random valid point in the workspace
        if p1 is None, make a random valid point in the workspace
        '''
        if p0 is None:
            p0 = self.rand_workspace_pt()
        if p1 is None:
            p1 = self.rand_workspace_pt()
        # MoveL(self, p0, p1, f, t)
        # my_traj = self.PathPlan.MoveL(p0, p1, f, t)




    # def MoveJ_rand(self):
    #     '''
    #     generate two valid, reachable points in workspace
    #     find the pose to reach the desired next point,
    #     interpolate joint variables to new pose over time ((q_vect, t), trajectory)
    #     '''
    #     pass

    def rand_workspace_pt(self):
        # generate a random q-vector within constraints of joint_angles
        # and return the pose if it is valid (not intersecting with itself)
        random_q_vect = np.zeros(len(self.rob.q_vect))
        for i, b in enumerate(self.rob.bounds):
            low_b, high_b = b
            random_q_vect[i] = np.random.uniform(low= low_b,
                                                 high= high_b,
                                                 size= 1)
        # set q_vect to random joint pose perform inverse kinematics
        self.rob.q_vect = random_q_vect 
        # forward kinematics of end-effector resulting from random joint pose
        rand_H_W_n = self.rob.FK()[-1]   

        return random_q_vect, rand_H_W_n



        