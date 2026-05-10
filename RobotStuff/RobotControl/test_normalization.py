'''
Normalization of inputs and outputs are very important, especially for a network with a tanh output bounded to [-1, 1]
'''
import torch
import numpy as np

from forward_kinematics import Robot_math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

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
    device=device,
)
robot.WT = robot.make_homogenous_transformation(yaw=0, pitch=0, roll=180, x=0, y=0, z=0)



def normalize_to_range(x, low, high):
        """
        Normalizes tensor x to the range [-1, 1] based on 
        provided low and high bound tensors.
        """
        denom = torch.clamp(high - low, min=1e-8)
        # Formula: 2 * (x - low) / (high - low) - 1
        return 2.0 * ((x - low) / (denom)) - 1.0

def get_normal_joint_value(q_vect: torch.Tensor) -> torch.Tensor:
    """Map q_vect into [-1, 1]"""
    return normalize_to_range(x= q_vect, low= robot.low_bounds, high=robot.high_bounds)

def get_normal_joint_vel(delta_q: torch.Tensor) -> torch.Tensor:
    """Map delta_q into [-1, 1]"""
    # largest negative velocity = moving from high to low in one step
    # largest positive velocity = moving from low to high in one step
    return normalize_to_range(x= delta_q, 
                              low= (robot.low_bounds - robot.high_bounds),
                              high= (robot.high_bounds - robot.low_bounds))


def get_normal_joint_acc(delta_q_prev: torch.Tensor,
                         delta_q_new: torch.Tensor) -> torch.Tensor:
        """Normalised change in normalised velocity, mapping joint acc to [-1, 1]"""
        Nv_prev = get_normal_joint_vel(delta_q_prev)
        Nv_new  = get_normal_joint_vel(delta_q_new)
        return (Nv_new - Nv_prev) / 2.0



def test_normalizing_inputs():

    '''
    my_bounds = [
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-180), np.deg2rad(0)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    ]   
    '''



    q_vect_1 = torch.tensor([np.deg2rad(-90), # -1
                             np.deg2rad(-90), # 0
                             np.deg2rad(-90), # -1
                             np.deg2rad(-90), # -1
                             np.deg2rad(-90), # -1
                             np.deg2rad(-90)]) # -1

    q_vect_2 = torch.tensor([np.deg2rad(-90), # -1
                             np.deg2rad(-180), # -1
                             np.deg2rad(0), # 0
                             np.deg2rad(90), # 1
                             np.deg2rad(90), # 1
                             np.deg2rad(90)]) # 1
    
    q_vect_3 = torch.tensor([np.deg2rad(-90), # -1
                             np.deg2rad(0), # 1
                             np.deg2rad(-90), # -1
                             np.deg2rad(-90), # -1
                             np.deg2rad(-90), # -1
                             np.deg2rad(-90)]) # -1
    
    delta_q_1 = q_vect_2 - q_vect_1
    delta_q_2 = q_vect_3 - q_vect_2

    print(f'normal joint value q_1 = {get_normal_joint_value(q_vect= q_vect_1)}')
    print(f'normal joint value q_2 = {get_normal_joint_value(q_vect= q_vect_2)}')
    print(f'normal joint value q_3 = {get_normal_joint_value(q_vect= q_vect_3)}')

    


    print(f'normal joint vel delta_q1 = {get_normal_joint_vel(delta_q= delta_q_1)}')
    print(f'normal joint vel delata_2 = {get_normal_joint_vel(delta_q= delta_q_2)}')




    print(f'normal joint acc delta_q2 - delata_q2 = {get_normal_joint_acc(delta_q_new= delta_q_2,
                                                                          delta_q_prev= delta_q_1)}')
    


if __name__ == "__main__":
      test_normalizing_inputs()