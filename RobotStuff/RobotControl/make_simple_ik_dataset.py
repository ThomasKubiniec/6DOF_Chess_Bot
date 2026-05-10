'''
Make a million or more random q_vectors (to pretty much guarantee workspace coverage),
get their raw XYZ coordinates and SO3 orientations.
'''
import torch
import numpy as np
from collections import deque
import pickle
import random

from forward_kinematics import Robot_math
from path_planning_math import PathPlannerMath


class data_generator:
    def __init__(self,
                 robot: Robot_math,
                 path_planner: PathPlannerMath,
                 goal_datapoints: int,
                 dataset_filename: str = "simple_ik_training_data"):

        self.robot = robot
        self.path_planner = path_planner
        self.goal_datapoints = int(goal_datapoints)          # guard against float (e.g. 1e7)
        self.dataset_filename = dataset_filename

        self.my_dataset = deque(maxlen=self.goal_datapoints)

    # ===========================================================
    # Making the dataset
    # ===========================================================
    def make_random_q_vect(self):
        return self.path_planner.make_random_poses()

    def get_coords(self, q_vect):
        self.robot.q_vect = q_vect
        return self.robot.give_ds()[-1], self.robot.give_Rs()[-1]

    def make_dataset(self):
        '''
        The dataset stores raw (position, SO3) pairs for random joint configs.
        Normalization of inputs is handled by the network at training/inference time.
        '''
        for i in range(self.goal_datapoints):
            coords = self.get_coords(q_vect=self.make_random_q_vect())  # FIX: was missing ()
            self.my_dataset.append(coords)

            if (i + 1) % 10_000 == 0:
                print(f"  Generated {i + 1:,} / {self.goal_datapoints:,} datapoints")

    # ------------------------------------------------------------------
    # Sampling / persistence
    # ------------------------------------------------------------------
    def sample(self, batchsize: int = 512):
        if len(self.my_dataset) < batchsize:
            return None
        return random.sample(list(self.my_dataset), batchsize)

    def save_dataset(self):
        path = f"{self.dataset_filename}.pkl"   # FIX: self.dataset_filename now defined
        with open(path, "wb") as f:
            pickle.dump(self.my_dataset, f)
        print(f"Saved {len(self.my_dataset):,} frames → {path}")

    def load_dataset(self, filename: str):
        path = f"{filename}.pkl"
        with open(path, "rb") as f:
            self.my_dataset = pickle.load(f)
        print(f"Loaded {len(self.my_dataset):,} frames ← {path}")


if __name__ == "__main__":
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

    path_planner = PathPlannerMath(my_robot= robot, dtype= torch.float64, device= device)

    my_data_gen = data_generator(robot= robot,
                                 path_planner= path_planner,
                                 goal_datapoints= int(1e6),
                                 dataset_filename= 'simple_ik_training_data')
    my_data_gen.make_dataset()
    my_data_gen.save_dataset()