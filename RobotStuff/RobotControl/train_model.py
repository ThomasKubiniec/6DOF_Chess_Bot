"""
Using a dataset, train an IK Neural Network and save its weights.

GPU-aware: training runs entirely on the detected device.
All tensors in the batch are moved to device before the forward pass.
"""

import torch
import torch.nn as nn
import numpy as np

from forward_kinematics import Robot_math
from rot_math import to_6D_R, to_SO3, YPR_SO3, DEVICE
from expert_inv_kin import Oracle
from loss_math import Loss_Math
from path_planning_math import PathPlannerMath
from ik_nn import IK
from make_dataset import my_dataframe, data_generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_robot(device: torch.device) -> Robot_math:
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
    robot.WT = robot.make_homogenous_transformation(
        yaw=0, pitch=0, roll=180, x=0, y=0, z=0)
    return robot


def batch_to_tensors(batch, device: torch.device):
    """
    Convert a list of (inputs_tuple, delta_q, target_weight) samples into
    a single batched tensor tuple ready for the network.

    inputs_tuple = (delta_q_N, q_N, dist_N, rot_6D)
    """
    delta_q_N_list, q_N_list, dist_N_list, rot_6D_list = [], [], [], []
    delta_q_out_list, tw_list = [], []

    for (inp, out, tw) in batch:
        delta_q_N, q_N, dist_N, rot_6D = inp
        delta_q_N_list.append(delta_q_N)
        q_N_list.append(q_N)
        dist_N_list.append(dist_N)
        rot_6D_list.append(rot_6D)
        delta_q_out_list.append(out)
        tw_list.append(torch.tensor(tw, dtype=torch.float64))

    def _stack(lst):
        return torch.stack(lst).to(dtype=torch.float64, device=device)

    X = torch.cat([_stack(delta_q_N_list),
                   _stack(q_N_list),
                   _stack(dist_N_list),
                   _stack(rot_6D_list)], dim=1)          # (B, input_dim)

    Y  = _stack(delta_q_out_list)                        # (B, n_joints)
    TW = _stack(tw_list).unsqueeze(1)                    # (B, 1)

    return X, Y, TW


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model: IK,
          dataset: my_dataframe,
          epochs: int = 50,
          batch_size: int = 512,
          lr: float = 1e-3,
          save_path: str = "ik_model.pt"):

    device    = model.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse       = nn.MSELoss(reduction='none')

    model.train()

    for epoch in range(epochs):
        batch = dataset.sample(batchsize=batch_size)
        if batch is None:
            print("Not enough data yet — skipping epoch.")
            continue

        X, Y, TW = batch_to_tensors(batch, device)

        optimizer.zero_grad()
        pred = model(X)                            # (B, n_joints)

        # Per-sample MSE weighted by target_weight
        loss_per_sample = mse(pred, Y).mean(dim=1, keepdim=True)  # (B, 1)
        loss = (loss_per_sample * TW).mean()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | loss = {loss.item():.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    robot  = build_robot(device)

    solver = Oracle(robot_class=robot,
                    pos_w=100, rot_w=5,
                    crash_w=2, dist_w=0.5,
                    vel_lambda=[0.1] * 6,
                    acc_lambda=[0.1] * 6)

    solver.L.max_good_err_pos = 0.1
    solver.L.max_good_err_deg = 2.0
    solver.L.max_ok_err_pos   = 0.25
    solver.L.max_ok_err_deg   = 6.0

    path_planner = PathPlannerMath(my_robot=robot)

    data_gen = data_generator(
        robot=robot, solver=solver, path_planner=path_planner,
        low_frac=1/2, high_frac=5/6,
        std_low=0.4, std_high=0.4,
        p_low=0.75,
    )

    dataset = my_dataframe(
        my_data_gen=data_gen,
        datapoints_goal=2e5,
        dataset_filename='training_data',
    )

    # ── Option A: generate dataset on the fly ────────────────────────
    # dataset.make_dataset()

    # ── Option B: load a pre-generated dataset ───────────────────────
    # dataset.load_dataset('training_data_5_6_26')

    model = IK(
        robot=robot,
        hid_dim=256,
        hid_layers=4,
    )

    train(
        model=model,
        dataset=dataset,
        epochs=200,
        batch_size=512,
        lr=1e-3,
        save_path="ik_model.pt",
    )