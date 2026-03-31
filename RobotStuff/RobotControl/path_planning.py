"""
Path Planning provides trajectories for the end-effector to get to one place to another.
MoveL interpolates a straight line between points.
MoveJ interpolates the joints directly to their new values. 
"""

import numpy as np
from forward_kinematics import Robot_math

class PathPlanner:
    def __init__(self, my_robot):
        pass
    

    def cubic_interp_comp(self, c0, c1, f, t):
        '''
        interpolates scalar c0 to c1 over f frames in timeframe t seconds.
        returns the trajectory of this scalar c0 to c1.
        '''
        pass

    def cubic_interp_vect(self, v0, v1, f, t):
        '''
        interpolates every component of v0 to v1 over f frames
        in a cubic trajectory, yeilding a smooth acceleration. 
        The components speed up and then slow down to reach the targets.
        '''
        pass


    def MoveL(self, p0, p1, f, t):
        '''
        If p0 and p1 are valid, reachable positions in the workspace,
        interpolate f number of frames between each component of the trajectory.
        
        Ex: p0 [0, 0, 0] -> p1 [1, 2, 3], f = 2:
        f_1 = [0.5, 1, 1.5], f_2 = [1, 2, 3]


        Do inverse kinematics for each point
        ''' 
        pass