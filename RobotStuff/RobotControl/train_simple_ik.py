'''
Train a network that predicts change in pose without imitation learning.
Loss is computed on FK coordinates of the predicted joint config vs the target.
'''

import torch
import torch.nn as nn
import numpy as np

from forward_kinematics import Robot_math
from make_simple_ik_dataset import data_generator
from simple_ik_model_maker import simpleIK
from path_planning_math import PathPlannerMath


# ---------------------------------------------------------------------------
# Robot construction
# ---------------------------------------------------------------------------
def build_robot(device: torch.device) -> Robot_math:
    my_a     = [0.0, 7.375, 0.0,  0.0,  0.0,  0.0]
    my_alpha = [np.deg2rad(90),  np.deg2rad(180), np.deg2rad(90),
                np.deg2rad(90),  np.deg2rad(-90),  np.deg2rad(0)]
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


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------
def batch_to_tensors(batch: list,
                     model: simpleIK,
                     data_gen: data_generator,
                     device: torch.device):
    '''
    batch  : list of (goal_pos (3,), goal_ori_SO3 (3,3)) sampled from the dataset
    Returns:
        q_starts    : (B, n)     random starting joint configs
        goal_xyz    : (B, 3)     target end-effector positions
        goal_SO3    : (B, 3, 3)  target end-effector orientations
        Y           : (B, 12)    target (xyz, flattened SO3) — used for loss
    '''
    B = len(batch)

    q_starts_list  = []
    goal_xyz_list  = []
    goal_SO3_list  = []

    for goal_pos, goal_ori_SO3 in batch:                    # FIX: iterate items, not range(batch)
        rand_q = data_gen.make_random_q_vect()

        q_starts_list.append(rand_q.to(torch.float32))
        goal_xyz_list.append(goal_pos.to(torch.float32))   # FIX: reassign .to() result
        goal_SO3_list.append(goal_ori_SO3.to(torch.float32))

    def _to_device(lst, extra_dims=None):
        return torch.stack(lst).to(dtype=torch.float32, device=device)

    q_starts = _to_device(q_starts_list)      # (B, n)
    goal_xyz = _to_device(goal_xyz_list)      # (B, 3)
    goal_SO3 = _to_device(goal_SO3_list)      # (B, 3, 3)

    # Build targets: (B, 12) = (xyz, flattened SO3)
    goal_SO3_flat = goal_SO3.reshape(B, -1)   # (B, 9)
    Y = torch.cat([goal_xyz, goal_SO3_flat], dim=1)   # (B, 12)

    return q_starts, goal_xyz, goal_SO3, Y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model: simpleIK,
          dataset: data_generator,
          epochs: int     = 1000,
          batch_size: int = 2048,
          lr: float       = 1e-3,
          save_path: str  = "simple_ik_model.pt"):

    device    = model.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse       = nn.MSELoss()

    # ── Sanity-check target distribution before training ────────────
    probe = dataset.sample(batchsize=4096)
    if probe is not None:
        _, goal_xyz_p, goal_SO3_p, Y_p = batch_to_tensors(probe, model, dataset, device)
        print("── Target distribution (Y) ──────────────────────")
        print(f"  mean : {Y_p.mean():.4f}")
        print(f"  std  : {Y_p.std():.4f}")
        print(f"  min  : {Y_p.min():.4f}")
        print(f"  max  : {Y_p.max():.4f}")
        print("─────────────────────────────────────────────────")

    model.train()

    for epoch in range(epochs):
        batch = dataset.sample(batchsize=batch_size)
        if batch is None:
            print("Not enough data yet — skipping epoch.")
            continue

        q_starts, goal_xyz, goal_SO3, Y = batch_to_tensors(
            batch, model, dataset, device)

        optimizer.zero_grad()

        # FIX: use batched entry point; model receives raw tensors and
        #      handles normalisation internally
        pred = model.give_IK_found_pose_batch(
            q_batch=q_starts,
            goal_xyz_batch=goal_xyz,
            goal_SO3_batch=goal_SO3,
        )   # (B, 12)

        loss = mse(pred, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | loss = {loss.item():.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    robot = build_robot(device)

    path_planner = PathPlannerMath(
        my_robot=robot,
        dtype=torch.float32,
        device=device,
    )

    data_gen = data_generator(
        robot=robot,
        path_planner=path_planner,
        goal_datapoints=int(1e7),              # FIX: int(), not float
        dataset_filename="simple_ik_training_data",
    )

    data_gen.load_dataset(filename="simple_ik_training_data")

    model = simpleIK(
        robot=robot,
        device=device,                         # FIX: device now passed explicitly
        hid_dim=256,
        hid_layers=4,
    )

    train(
        model=model,
        dataset=data_gen,
        epochs=int(1e5),
        batch_size=2048,
        lr=1e-4,
        save_path=f"simp_ik_model_hdim{model.hid_dim}_hdep{model.hid_layers}.pt",
    )