import numpy as np
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Heuristic policy using sonar sensors.
    """

    left = np.any(obs[0:4])
    front = np.any(obs[4:12])
    right = np.any(obs[12:16])

    ir = obs[16]
    stuck = obs[17]

    if stuck:
        return ACTIONS[int(rng.choice([0, 2, 4]))]

    if ir:
        return "FW"

    if front:
        return "FW"

    if left:
        return "L22"

    if right:
        return "R22"

    # Exploration if none of the sensors fire up
    # Mostly move forward, sometimes rotate
    probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    action_index = rng.choice(len(ACTIONS), p=probs)

    return ACTIONS[int(action_index)]