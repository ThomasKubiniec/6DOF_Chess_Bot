"""
DQN / MLP network factory.

Used in TD3 as:
  • Actor    (state → joint delta, output activation = Tanh)
  • Critic   (state‖action → scalar value, output activation = Identity/Linear)

Architecture:
  Linear(in, hid) → ReLU
  [ Linear(hid, hid) → ReLU ] × hidden_depth
  Linear(hid, out) → output_activation

Bugs fixed vs original:
  1. DQN now inherits nn.Module so .parameters(), .to(), state_dict() etc. work.
  2. output_function must be an instantiated nn.Module, not a class.
     (Callers previously passed nn.Tanh / nn.Linear; now pass nn.Tanh() / nn.Identity().)
  3. Removed the erroneous dtype=torch.float32 `.to()` call that silently
     downcast float64 robot tensors to float32 — dtype is now left as the
     default (float32) but clearly documented.  If you need float64 throughout,
     pass dtype=torch.float64 in the call below.
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Fully-connected neural network with configurable depth and output activation.

    Parameters
    ----------
    input_size        : number of input features
    output_size       : number of output features
    hidden_width      : neurons per hidden layer
    hidden_depth      : number of *additional* hidden layers (beyond the first)
    output_activation : an instantiated nn.Module applied after the final Linear,
                        e.g. nn.Tanh() for the actor, nn.Identity() for critics.
    dtype             : torch dtype for all parameters (default torch.float32;
                        use torch.float64 to match the robot's FK tensors)
    """

    def __init__(self,
                 input_size:        int,
                 output_size:       int,
                 hidden_width:      int,
                 hidden_depth:      int,
                 output_activation: nn.Module,
                 dtype:             torch.dtype = torch.float32):
        super().__init__()

        self.in_size  = input_size
        self.out_size = output_size
        self.hid_w    = hidden_width
        self.hid_d    = hidden_depth
        self.dtype    = dtype

        # ── Build layer list ────────────────────────────────────────────────
        layers = []

        # Input → first hidden
        layers.append(nn.Linear(input_size, hidden_width))
        layers.append(nn.ReLU())

        # Additional hidden → hidden blocks
        for _ in range(hidden_depth):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())

        # Final hidden → output + activation
        layers.append(nn.Linear(hidden_width, output_size))
        layers.append(output_activation)

        self.net = nn.Sequential(*layers)

        # ── Move to device & dtype ──────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)