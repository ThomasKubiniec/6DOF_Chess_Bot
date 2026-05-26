"""
Rotation math utilities - pure PyTorch implementation.
Covers SO3 rotation matrices, 6D rotation representation, and Euler angle constructors.

GPU-aware: every function accepts an optional `device` argument (defaults to
cuda if available, cpu otherwise).  Pass `device` explicitly when mixing
devices to avoid cross-device errors.
"""
import torch
import math

# ---------------------------------------------------------------------------
# Module-level default device — set once, used everywhere
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Rx_SO3(theta_x_deg: float, device=None) -> torch.Tensor:
    """Rotation matrix about the X axis."""
    device = device or DEVICE
    t = math.radians(theta_x_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ], dtype=torch.float64, device=device)


def Ry_SO3(theta_y_deg: float, device=None) -> torch.Tensor:
    """Rotation matrix about the Y axis."""
    device = device or DEVICE
    t = math.radians(theta_y_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([
        [ c,  0,  s],
        [ 0,  1,  0],
        [-s,  0,  c],
    ], dtype=torch.float64, device=device)


def Rz_SO3(theta_z_deg: float, device=None) -> torch.Tensor:
    """Rotation matrix about the Z axis."""
    device = device or DEVICE
    t = math.radians(theta_z_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1],
    ], dtype=torch.float64, device=device)


def YPR_SO3(yaw_deg: float, pitch_deg: float, roll_deg: float,
            device=None) -> torch.Tensor:
    """
    Yaw-Pitch-Roll (ZYX) rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    device = device or DEVICE
    return Rz_SO3(yaw_deg, device) @ Ry_SO3(pitch_deg, device) @ Rx_SO3(roll_deg, device)


def to_6D_R(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3x3 SO3 rotation matrix to the 6D representation
    (first two columns of R, flattened): shape (6,)
    """
    R = R.to(torch.float64)
    return torch.cat([R[:, 0], R[:, 1]])  # (6,)  — stays on same device as R


def to_6D_R_batch(R: torch.Tensor) -> torch.Tensor:
    """
    Batched version of to_6D_R.

    R   : (B, 3, 3)
    returns : (B, 6)  — first two columns of each matrix, concatenated
    """
    R = R.to(torch.float64)
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)  # (B, 6)


def to_SO3_batch(r6d: torch.Tensor) -> torch.Tensor:
    """
    Batched version of to_SO3.  Gram-Schmidt orthonormalisation over a batch.

    r6d : (B, 6)
    returns : (B, 3, 3)
    """
    r6d = r6d.to(torch.float64)
    a1 = r6d[:, :3]   # (B, 3)
    a2 = r6d[:, 3:6]  # (B, 3)

    b1 = a1 / torch.linalg.vector_norm(a1, dim=1, keepdim=True)           # (B, 3)
    dot = (b1 * a2).sum(dim=1, keepdim=True)                               # (B, 1)
    b2 = a2 - dot * b1
    b2 = b2 / torch.linalg.vector_norm(b2, dim=1, keepdim=True)           # (B, 3)
    b3 = torch.linalg.cross(b1, b2, dim=1)                                # (B, 3)

    return torch.stack([b1, b2, b3], dim=2)  # (B, 3, 3)  — columns of R


def to_SO3(r6d: torch.Tensor) -> torch.Tensor:
    """
    Recover a 3x3 SO3 rotation matrix from a 6D representation.
    Uses Gram-Schmidt orthonormalisation on the two stored columns.
    Output lives on the same device as the input.
    """
    r6d = r6d.to(torch.float64)
    a1 = r6d[:3]
    a2 = r6d[3:6]

    b1 = a1 / torch.linalg.vector_norm(a1)
    b2 = a2 - torch.dot(b1, a2) * b1
    b2 = b2 / torch.linalg.vector_norm(b2)
    b3 = torch.linalg.cross(b1, b2)

    return torch.stack([b1, b2, b3], dim=1)  # (3, 3)