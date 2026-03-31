"""
Sources:
https://zhouyisjtu.github.io/project_rotation/cvpr_poster.pdf

https://openreview.net/pdf?id=RFjhxXrTlX#:~:text=The%20so%2Dcalled%20'6D%20representation,large%20gradients%20that%20destabilize%20training.&text=f!&text=Matrix%20'(!


Converting SO3 R (3x3) (9D), to 6D and back to save input neurons 


SO3_R = 
[[a1].T, [a2].T, [a3].T]

g_GS(SO3_R) = [[a1].T, [a2].T] (6D_R) (Stiefel Manifold)

f_GS(6D_R) = [[b1].T, [b2].T, [b3].T] (Grant Schmidt Re-orthogonalization)
b1 = N(a1)
b2 = N(a2 - (b1 @ a2) * b1) # b1 dot a2
b3 = b1 X b2 (cross product)

N(v) = v / norm(v) # Normalization
"""

import numpy as np
import torch

def N(v):
    """Normalization a vector v"""
    return v / torch.linalg.norm(v)


def g_GS(SO3_R):
    """
    Convert a 3x3 rotation matrix to a 6D representation
    by extracting the first two columns.
    Saves space for input neurons while providing continuous manifold and minimal loss of info
    """
    a1 = SO3_R[:, 0] # first column
    a2 = SO3_R[:, 1] # second column

    # _6D_R = np.column_stack((a1, a2))
    
    # flatten into a 1D array
    _6D_R_vect = torch.tensor(np.concatenate((a1, a2)))
    return _6D_R_vect


def f_GS(_6D_R_vect):
    """
    Convert a flattend 6D representation of SO3 3x3 rotation matrix
    back to SO3 3x3 R by Grant-Schmidt-like-orthogonalization
    """
    a1 = _6D_R_vect[:3]
    a2 = _6D_R_vect[3:]
    
    b1 = N(a1)
    b2 = N(a2 - (b1 @ a2) * b1)
    b3 = torch.cross(b1, b2)

    SO3_R = torch.column_stack((b1, b2, b3))
    return SO3_R