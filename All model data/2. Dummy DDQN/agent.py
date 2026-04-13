"""
Submission agent for OBELIX using trained DDQN weights.
Place weights.pth in the same directory as this file.
"""

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None


class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def _load_once():
    """Load the trained model once."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py")

    model = DQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return best action predicted by the trained network."""
    _load_once()

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0).numpy()

    action_index = int(np.argmax(q_values))
    return ACTIONS[action_index]