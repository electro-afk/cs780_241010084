import numpy as np
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

# Persistent memory
_search_counter = 0
_escape_dir = None
_escape_toggle = False
_last_obs = None


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _search_counter, _escape_dir, _escape_toggle, _last_obs

    # Detect new episode
    if _last_obs is None or np.sum(obs) == 0:
        _escape_dir = None
        _escape_toggle = False

    _last_obs = obs.copy()

    # Sensor groups
    left = np.any(obs[0:4])
    front = np.any(obs[4:12])
    right = np.any(obs[12:16])
    ir = obs[16]
    stuck = obs[17]

    # Push
    if ir:
        return "FW"

    # Stuck
    if stuck:

        # choose escape direction once per episode
        if _escape_dir is None:
            _escape_dir = "L45" if rng.random() < 0.5 else "R45"

        # alternate rotate → forward
        _escape_toggle = not _escape_toggle

        if _escape_toggle:
            return _escape_dir
        else:
            return "FW"

    # Approach
    if front:
        return "FW"

    # Align
    if left and not right:
        return "L22"

    if right and not left:
        return "R22"

    if left and right:
        return "FW"

    # Find
    _search_counter += 1

    # sweeping exploration
    if _search_counter % 20 < 15:
        return "FW"

    # occasional turn to change direction
    if rng.random() < 0.5:
        return "L22"
    else:
        return "R22"