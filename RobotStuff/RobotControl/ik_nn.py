"""
Imitation Learning based Inverse Kinematics.
Distill the iterative optimizer based solutions into one forward pass calculation to save time.
"""

import torch
import torch.nn as nn

from forward_kinematics import Robot_math # the math that simulates the robot moving
from path_planning_math import PathPlannerMath # the math that tells the robot where to move
from expert_inv_kin import Oracle # the math that tells the robot how to move given where to move
from loss_math import Loss_Math # math that lets us normalize inputs and rescale normalized outputs





class IK(nn):
    def __int__(self,
                robot : Robot_math, hid_dim, hid_layers,
                pos_w=None, rot_w=None, crash_w=None, dist_w=None, vel_lambda=None, acc_lambda=None):
        super.__init__()
        
        self.robot = robot

        self.L = Loss_Math(my_robot= self.robot,
                           pos_w= pos_w, rot_w= rot_w, crash_w= crash_w, dist_w= dist_w,
                           vel_lambda= vel_lambda, acc_lambda= acc_lambda)


        self.n = len(self.robot.a)
        self.input_dim = (2 * self.n) + (3) + (6) # 2 x joints + xyz coords + 6 dim orientation
        self.output_dim = self.n # just the joints being controlled


        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(nn.Linear(self.input_dim, hid_dim))

        # hidden layers
        for hid_layer in range(hid_layers):
            self.layers.append(nn.Linear(hid_dim, hid_dim))

        # output layer
        self.layers.append(nn.Tanh(hid_dim, self.output_dim)) # automatic [-1, 1] normalization of output


    def forward(self, x):
        '''
        input: A tensor consisting of:
            - previous joint velocity normalized
            - current joint pose normalized
            - distance from goal to end-effector normalized
            - 6D goal orientation (already normalized) 

        output: A tensor consisting of 
            - the normalized joint velocity to get to the goal position and orientation
        '''
        for layer in self.layers:
            x = layer(x)
        return x
    


    def solve_IK(self):
        '''
        convert unnormalized inputs to normalized inputs, then forward to get normalized output

        rescale output, add to unnormalized current joint pose, and return
        '''
        pass
