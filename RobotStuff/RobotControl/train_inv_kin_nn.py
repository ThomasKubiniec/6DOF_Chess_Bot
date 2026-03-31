import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forward_kinematics import Robot_math
from rot_math import g_GS, f_GS, N


'''
I could try encoding joint velocity explicitly instead of implicitly:
implicit: (q_t_-_1, q_t)
explicit: v_t = (q_t - q_t_-_1)
'''

"""
This is essentially an unrolled CNN. 
We provide the previous joint velocity to change how we would move
compared to moving from a stand-still.

In a CNN, we would pass the entire trajectory we want to go through at once. 
"""




class IKNet(nn.Module):
    """
    This network takes in the previous and current joint pose,
    the desired end-effector position and orientation (in 6D rotation representation),
    and outputs the change in joint pose that 
    will move the end-effector to the goal when summed with the current joint pose.
    """

    def __init__(self,
                 my_robot, # Robot_math class object
                 hidden_layers= 3, 
                 width_list= [128] * 5, # how many neurons are in each hidden layer
                 activ_fn= F.leaky_relu):
        
        self.robot = my_robot
        self.joints= len(self.robot.a) # number of joints in the robot
        self.input_width= (self.joints * 2) + 3 + 6 # q_prev, q_curr, G_d, G_6D_R
        self.output_width= self.joints # outputs delta q, change in joint pose to move to goal

        super(IKNet, self).__init__()
        self.activ_fn = activ_fn
        # VVV------------------ create layers ------------------VVV
        self.input_layer = nn.Linear(self.input_width, width_list[0])
        self.my_layers = [self.input_layer]

        for i in range(len(hidden_layers) - 1):
            setattr(self, f"hl{i}", nn.Linear(width_list[i], width_list[i + 1]))
            self.my_layers.append(getattr(self, f"hl{i}"))
        
        self.output_layer = nn.Linear(width_list[-1], self.output_width)
        self.my_layers.append(self.output_layer)
        # ^^^------------------ create layers ------------------^^^



    def forward(self, x):
        for layer in self.my_layers:
            x = self.activ_fn(layer(x))
        return x
            

    def L(self, q_prev, q_curr, G_d, G_6D_R, q_next_pred):
        """
        Loss function.
        Weights MSE loss of accuracy, change in joint pose, and change in joint velocity
        """
        lam_pos = 1.0
        lam_rot = 0.1
        lam_v = [0.1] * self.joints
        lam_a = [0.1] * self.joints

        my_e_p = self.err_pos(q_curr, q_next_pred, G_d, scale= lam_pos) # position accuracy
        my_e_r = self.err_rot(q_curr, q_next_pred, G_6D_R, scale= lam_rot) # orientation accuracy
        my_e_v = self.err_vel(q_next_pred, scale_vect= lam_v) # change in joint pose
        my_e_a = self.err_acc(q_prev, q_curr, q_next_pred, scale= lam_a) # change in joint velocity
        return my_e_p + my_e_r + my_e_v + my_e_a

        


    def err_pos(self, q_curr, q_next_pred, G_d, scale= 1.0):
        """
        Positional error between the end-effector and the desired position
        """
        self.robot.q_vect = q_curr + q_next_pred # predicted IK pose
        # runs forward kinematics and extracts the positions of the joints in the world frame
        ee_pos = self.robot.give_ds()[-1]  
        return scale * torch.norm(ee_pos - G_d)


    def err_rot(self, q_curr, q_next_pred, G_6D_R, scale= 0.1):
        """
        Rotational error between the end-effector and the desired orientation
        """

        self.robot.q_vect = q_curr + q_next_pred # predicted IK pose
        # runs forward kinematics and extracts the orientations of the joints in the world frame
        ee_R = self.robot.give_Rs()[-1]

        SO3_R = f_GS(G_6D_R)
        # print(f"SO3_R = \n{SO3_R}")
        # print(f"ee_R = \n{ee_R}")

        # Frobenius norm of the difference between the predicted and desired rotation matrices
        # print(f"frobenius norm error = {torch.from_numpy(ee_R - SO3_R)}")
        return scale * torch.norm(torch.from_numpy(ee_R - SO3_R))



    def err_vel(self, q_next_pred, scale_vect= None):
        """
        Change in joint pose (velocity) should be small to 
        encourage smooth movements to get to the goal.
        
        The network predicts change in joint pose, 
        so the predicted pose is q_curr + q_next_pred
        """
        if scale_vect is None:
            scale_vect = [0.1] * self.joints

        # scale each joint with a unique weight to 
        # dicourage movements in some joints more than others
        lam_v = torch.diag(scale_vect)
        return torch.norm(lam_v @ q_next_pred)
    

    def err_acc(self, q_prev, q_curr, q_next_pred, scale_vect=None):
        """
        Change in joint velocity (acceleration) should be small to discourage jerky movements.
        q_next_pred is the change in joint pose, 
        so q_curr + q_next_pred is the predicted next pose
        
        If the acceleration is very large, 
        then the robot is making jerky movements during path planning, which is undesirable. 
        """
        if scale_vect is None:
            scale_vect = [0.1] * self.joints

        # scale each joint with a unique weight to 
        # dicourage movements in some joints more than others
        lam_a = torch.diag(scale_vect)
        return torch.norm(lam_a @ ((q_next_pred + q_curr) - 2 * q_curr + q_prev))






def train(train_episodes,
          learning_rate):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    IKNet.to(device)