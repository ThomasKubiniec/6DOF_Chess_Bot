"""
Rotation math utilities - pure PyTorch implementation.
Covers SO3 rotation matrices, 6D rotation representation, and Euler angle constructors.
"""
import torch
import math


def Rx_SO3(theta_x_deg: float) -> torch.Tensor:
    """Rotation matrix about the X axis."""
    t = math.radians(theta_x_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ], dtype=torch.float64)


def Ry_SO3(theta_y_deg: float) -> torch.Tensor:
    """Rotation matrix about the Y axis."""
    t = math.radians(theta_y_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([
        [ c,  0,  s],
        [ 0,  1,  0],
        [-s,  0,  c],
    ], dtype=torch.float64)


def Rz_SO3(theta_z_deg: float) -> torch.Tensor:
    """Rotation matrix about the Z axis."""
    t = math.radians(theta_z_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1],
    ], dtype=torch.float64)


def YPR_SO3(yaw_deg: float, pitch_deg: float, roll_deg: float) -> torch.Tensor:
    """
    Yaw-Pitch-Roll (ZYX) rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    return Rz_SO3(yaw_deg) @ Ry_SO3(pitch_deg) @ Rx_SO3(roll_deg)


def to_6D_R(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3x3 SO3 rotation matrix to the 6D representation
    (first two columns of R, flattened): shape (6,)
    """
    R = R.to(torch.float64)
    return torch.cat([R[:, 0], R[:, 1]])  # (6,)


def to_SO3(r6d: torch.Tensor) -> torch.Tensor:
    """
    Recover a 3x3 SO3 rotation matrix from a 6D representation.
    Uses Gram-Schmidt orthonormalisation on the two stored columns.
    """
    r6d = r6d.to(torch.float64)
    a1 = r6d[:3]
    a2 = r6d[3:6]

    b1 = a1 / torch.linalg.vector_norm(a1)
    b2 = a2 - torch.dot(b1, a2) * b1
    b2 = b2 / torch.linalg.vector_norm(b2)
    b3 = torch.linalg.cross(b1, b2)

    return torch.stack([b1, b2, b3], dim=1)  # (3, 3)