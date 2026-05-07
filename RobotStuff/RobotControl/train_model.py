'''
Using a dataset, train an IK Neural Network and save its weights.
'''


import torch
import numpy as np

from forward_kinematics import Robot_math
from rot_math import to_6D_R, to_SO3, YPR_SO3
from expert_inv_kin import Oracle
from loss_math import Loss_Math
from path_planning_math import PathPlannerMath
from ik_nn import IK
from make_dataset import my_dataframe, data_generator


