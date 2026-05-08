'''
The goal is to make approximately 100k total samples of good and okay frames (trajectory points), 
discarding bad or crashing frames.

Good points have a target weight of 1.0. Okay has a target weight of 0.3. 
Loss is multiplied by the target_weight. 



Since this is a medium sized dataset, it will take a long
time to compute, so we will need to save it. 

The format of the dataset:
(q_curr, delta_q_prev, Goal_XYZ, Goal_Ori_SO3, delta_q_pred, target_weight)
'''

import torch
import numpy as np

import pickle
from collections import deque
import random

from forward_kinematics import Robot_math
from path_planning_math import PathPlannerMath
from expert_inv_kin import Oracle
from rot_math import YPR_SO3, to_SO3, to_6D_R



class data_generator:
    def __init__(self,
                 robot : Robot_math,
                 solver : Oracle,
                 path_planner : PathPlannerMath,
                 low_frac,
                 high_frac,
                 std_low,
                 std_high,
                 p_low):
        
        self.robot = robot
        self.solver = solver
        self.path_planner = path_planner

        self.mean_low = self.robot.max_reach * low_frac
        self.std_low = std_low

        self.mean_high = self.robot.max_reach * high_frac
        self.std_high = std_high
        
        self.p_low = p_low



    def make_random_unit_vect(self):
        rand_vect = torch.randn(3)
        rand_unit_vect = torch.nn.functional.normalize(input= rand_vect,
                                                       p=2,
                                                       dim= 0)
        return rand_unit_vect


    def make_random_radius(self,
                           rad_mean, rad_std):
        min_r = 0
        max_r = self.robot.max_reach

        rand_r = torch.nn.init.trunc_normal_(tensor= torch.empty(1), 
                                            mean= rad_mean, 
                                            std= rad_std, 
                                            a= min_r, 
                                            b= max_r)
        
        return rand_r



    def make_rand_xyz(self):
        '''
        Make a Bimodal distrobution of radii within the workspace 
        and some more clustered at the extremes of the workspace.
        The random xyz is made with a random unit vector multiplied by this scalar radius. 
        '''

        p = torch.rand(1)
        rand_u = self.make_random_unit_vect()

        if self.p_low <= p:
            rand_r = self.make_random_radius(rad_mean= self.mean_low,
                                             rad_std= self.std_low)

            return rand_u * rand_r

        else:
            rand_r = self.make_random_radius(rad_mean= self.mean_high,
                                             rad_std= self.std_high)

            return rand_u * rand_r


    def make_random_YPR(self):
        dist = torch.distributions.Uniform(low= -180.0, high= 180.0)

        sample = dist.sample(torch.zeros(3).shape)

        my_R = YPR_SO3(yaw_deg= sample[0].item(),
                       pitch_deg= sample[1].item(),
                       roll_deg= sample[2].item())
        
        return my_R



    def make_random_q_vect(self):
        valid = False
        while valid == False: # repeat until a valid point is found
            goal_xyz = self.make_rand_xyz()
            goal_ori_SO3 = self.make_random_YPR()

            # formatted to torch tensor inside get_IK_traditional()
            q_vect = self.solver.get_IK_traditional(Goal_Posi= goal_xyz, Goal_ori_SO3= goal_ori_SO3) 
            
            
            # append all non crashing trajectories
            # frames labeled 'bad' were made by the expert, and are considered necessary to learn,
            # but are learned with a 5% discount. 
            if solver.L.crashed == False:
                return q_vect 
            
            


    def normalize_inputs(self,
                         delta_q_prev, q_vect, 
                         target_xyz, target_ori):
        '''
        Turns real units of: 
        initial joint velocity (previous joint velocity to get current pose)
        current pose
        goal position
        goal orientation

        To ->
        joint velocity normalized by portion of joint range swept
        current pose normalized by joint range
        distance to target normalized by max reach
        difference in orientation represented in 6D

        This is the normalization that will be fed into the network. 
        This code should be copied to the network's methods 
        so the network can work with raw units when training is done. 
        
        It is done here to speed up inference time on training data. Do it once, not thousands of times.
        '''
        
        delta_q_N = self.solver.L.get_normal_joint_vel(delta_q= delta_q_prev)
        q_vect_N = self.solver.L.get_normal_joint_value(q_vect= q_vect)


        self.robot.q_vect = q_vect
        current_position = self.robot.give_ds()[-1]
        dist_to_targ_N = self.solver.L.get_normal_dist_to_goal(pos_curr= current_position,
                                                               pos_G_ws= target_xyz)

        current_ori_SO3 = self.robot.give_Rs()[-1]
        targ_ori_SO3 = to_SO3(target_ori)
        rot_to_targ_SO3 = targ_ori_SO3.T @ current_ori_SO3 # rotation from curr to goal in SO3
        rot_to_targ_6D_R = to_6D_R(rot_to_targ_SO3)
        
        return (delta_q_N, q_vect_N, dist_to_targ_N, rot_to_targ_6D_R)




    def make_random_traj(self):
        '''
        Make a random trajectory using random workspace coordinates instead of random joint poses.
        '''
        q1 = self.make_random_q_vect()
        q2 = self.make_random_q_vect()

        frame_low = 20
        frame_high = 100
        frame_steps = frame_high - frame_low + 1


        # pick a whole number of frames in a range. 
        frames_range = torch.linspace(start= frame_low, end= frame_high, steps= frame_steps)
        random_idx = torch.randint(0, len(frames_range), (1,)).item()
        frames = int(frames_range[random_idx].item())
    
        workspace_traj, time_delay, initial_pose = self.path_planner.MoveL(tot_time=1, 
                                                                           frames= frames,
                                                                           q_init= q1,
                                                                           q_end= q2)


        self.solver.follow_trajectory(traj_t_delay_start_q= (workspace_traj, time_delay, initial_pose),
                                      Print_Bool= False)

        if self.solver.L.crashed == True:
            return # failed trajectory 
        

        # We want:
        # Input = (delta_q_prev_N, q_curr_N, dist_to_goal_N, goal_ori_6D, target_weight)
        # Output = (delta_q_new_N)

        inputs = []
        outputs = []
        target_ws = []

        # print(f'workspace_traj[0] = {workspace_traj[0]}')
        # print(f'targ_pos = {workspace_traj[0][:3]}')
        # print(f'targ_ori = {workspace_traj[0][3:]}')

        # append the first traj point data once without an if-statement enhance speed of loop.
        inputs.append(self.normalize_inputs(delta_q_prev= torch.zeros(len(self.robot.a)),
                                            q_vect= initial_pose,
                                            target_xyz= workspace_traj[0][ : 3],
                                            target_ori= workspace_traj[0][3 : ]))
        
        outputs = [self.solver.current_trajectory[0][0]] # (delta_q_new, q_new)

        for i, ws_traj in enumerate(workspace_traj[1:]):
            inputs.append(self.normalize_inputs(delta_q_prev= self.solver.current_trajectory[i - 1][0],
                                                q_vect= self.solver.current_trajectory[i - 1][1],
                                                target_xyz= ws_traj[:3],
                                                target_ori= ws_traj[3:]))
            
            outputs.append(self.solver.current_trajectory[i][0]) # (delta_q_new, q_new)

        target_ws = self.solver.current_traj_grade.copy()
        # inputs, outputs, weighted importance
        return inputs, outputs, target_ws



class my_dataframe:
    def __init__(self, my_data_gen : data_generator, datapoints_goal, dataset_filename):
        self.my_data_gen = my_data_gen
        
        self.datapoints_goal = datapoints_goal
        self.my_dataset = deque(maxlen= round(self.datapoints_goal*1.05)) # give some clearance
        
        self.dataset_filename = dataset_filename

    def make_dataset(self):
        done = False
        trajectory_collected = 0
        while done == False:
            # self.my_dataset.append(self.my_data_gen.make_random_traj()) # inputs, outputs, weighted importance
            for traj_frame in self.my_data_gen.make_random_traj():
                self.my_dataset.append(traj_frame)

            trajectory_collected += 1
            
            if len(self.my_dataset) >= self.datapoints_goal:
                done = True
            
            print(f'collected {trajectory_collected} datapoints')

    

    def sample(self, batchsize= 512):
        if len(self.my_dataset) < batchsize:
            return None
        # return a batchsize number of input output pairs from the Expert generated dataset
        return random.sample(self.my_dataset, batchsize) 
        

    def save_dataset(self):
        # Saving to a file
        with open(f'{self.dataset_filename}.pkl', 'wb') as f:
            pickle.dump(self.my_dataset, f)



    def load_dataset(self, filename):
        # Loading back from the file
        with open(f'{filename}.pkl', 'rb') as f:
            self.my_dataset = pickle.load(f)




if __name__ == "__main__":
    
    # initialize 

    my_a     = [0.0, 7.375, 0.0,  0.0,  0.0,  0.0]
    my_alpha = [np.deg2rad(90),  np.deg2rad(180), np.deg2rad(90),
                np.deg2rad(90),  np.deg2rad(-90), np.deg2rad(0)]
    my_d     = [-3.5, 0.0, 0.0, 8.25, 0.0, 5.1875]
    my_theta = [np.deg2rad(0),   np.deg2rad(0),   np.deg2rad(90),
                np.deg2rad(180), np.deg2rad(0),   np.deg2rad(-90)]
    my_bounds = [
        (np.deg2rad(-90),  np.deg2rad(90)),
        (np.deg2rad(-180), np.deg2rad(0)),
        (np.deg2rad(-90),  np.deg2rad(90)),
        (np.deg2rad(-90),  np.deg2rad(90)),
        (np.deg2rad(-90),  np.deg2rad(90)),
        (np.deg2rad(-90),  np.deg2rad(90)),
    ]

    robot = Robot_math(
        a=my_a, alpha=my_alpha, d=my_d, theta=my_theta,
        joint_type=["r"] * 6, bounds=my_bounds,
        fail_dist=[0.1] * 6,
    )
    robot.WT = robot.make_homogenous_transformation(
        yaw=0, pitch=0, roll=180, x=0, y=0, z=0)

    solver = Oracle(robot_class=robot,
                    pos_w= 100,
                    rot_w= 5,
                    crash_w= 2,
                    dist_w= 0.5,
                    vel_lambda= [0.1] * 6,
                    acc_lambda= [0.1] * 6)


    path_planner = PathPlannerMath(my_robot=robot)


    data_gen = data_generator(robot= robot,
                            solver= solver,
                            path_planner= path_planner,
                            low_frac= 1/2, high_frac= 5/6,
                            std_low= 0.4, std_high= 0.4, p_low= 0.75)



    My_Dataset = my_dataframe(my_data_gen= data_gen,
                              datapoints_goal= 2e5,
                              dataset_filename= 'training_data_5_6_26')
    
    My_Dataset.make_dataset()