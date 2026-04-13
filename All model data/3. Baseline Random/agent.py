import numpy as np
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Random baseline policy.
    Mostly moves forward with occasional rotations.
    """

    probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05], dtype=float)

    action_index = rng.choice(len(ACTIONS), p=probs)

    return ACTIONS[int(action_index)]