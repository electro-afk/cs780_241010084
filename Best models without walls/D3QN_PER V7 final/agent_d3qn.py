"""
D3QN Agent — Submission file for OBELIX RL Challenge
=====================================================
Trained with: D3QN-PER (Dueling Double DQN + Prioritized Experience Replay)

PER infrastructure is training-only — completely absent here.

All other policy() logic (frame stacking, episode reset detection,
weight loading) is identical to agent_dqn.py.

Evaluate locally:
  python evaluate.py --agent_file agent_d3qn.py --difficulty 0
  python evaluate.py --agent_file agent_d3qn.py --difficulty 0 --wall_obstacles
"""

import os
from collections import deque
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
NUM_ACTIONS = len(ACTIONS)
OBS_DIM     = 18
STACK_K     = 4
INPUT_DIM   = OBS_DIM * STACK_K   # 72

WEIGHTS_FILE = "weights_d3qn_final.pth"


# ══════════════════════════════════════════════════════════════════════════════
# Dueling Network
# ══════════════════════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        return V + (A - A.mean(dim=1, keepdim=True))


# ══════════════════════════════════════════════════════════════════════════════
# Global agent state — persists across policy() calls within one episode
# ══════════════════════════════════════════════════════════════════════════════
_STATE: dict = {
    "model":        None,
    "frames":       None,
    "prev_obs":     None,
    "step":         0,
    "was_done":     False,
    "prev_stuck":   0,
    "prev_attach":  False,
}


def _load_model() -> DuelingDQN:
    if _STATE["model"] is not None:
        return _STATE["model"]

    weights_path = os.path.join(os.path.dirname(__file__), WEIGHTS_FILE)
    model = DuelingDQN(INPUT_DIM, NUM_ACTIONS)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        print(f"[agent_d3qn] Loaded weights from {weights_path}")
    else:
        model.eval()
        print(f"[agent_d3qn] WARNING: {weights_path} not found. Using random weights.")

    _STATE["model"] = model
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Episode reset detection
# ══════════════════════════════════════════════════════════════════════════════
def _is_new_episode(obs: np.ndarray) -> bool:
    """
    Detects episode boundaries without an explicit reset signal.

    Triggers on:
      1. First ever call
      2. Previous step was marked done
      3. Attachment was active then all sensors went silent (post-success reset)
      4. Stuck flag cleared AND sensors went from active to silent (teleport/reset)
    """
    if _STATE["prev_obs"] is None:
        return True
    if _STATE["was_done"]:
        return True

    prev          = _STATE["prev_obs"]
    all_silent    = not np.any(obs[:17])
    had_signal    = np.any(prev[:17])

    # Post-success: attachment was active, now everything is silent
    if _STATE["prev_attach"] and all_silent:
        return True

    # Teleport/reset: stuck cleared + sensors dropped
    if _STATE["prev_stuck"] > 0 and obs[17] == 0 and all_silent and had_signal:
        return True

    return False


def _init_episode(obs: np.ndarray) -> np.ndarray:
    _STATE["frames"]      = deque(maxlen=STACK_K)
    for _ in range(STACK_K):
        _STATE["frames"].append(obs.copy())
    _STATE["step"]        = 0
    _STATE["was_done"]    = False
    _STATE["prev_attach"] = False
    return np.concatenate(list(_STATE["frames"]), axis=0)


def _update_stack(obs: np.ndarray) -> np.ndarray:
    _STATE["frames"].append(obs.copy())
    return np.concatenate(list(_STATE["frames"]), axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Choose an action from the current 18-bit observation.

    Parameters
    ----------
    obs : np.ndarray, shape (18,)
        Current sensor reading from the environment.
    rng : np.random.Generator
        RNG provided by the evaluator (unused at inference — greedy policy).

    Returns
    -------
    str — one of "L45", "L22", "FW", "R22", "R45"
    """
    model = _load_model()

    # Episode boundary handling
    if _is_new_episode(obs):
        stacked = _init_episode(obs)
    else:
        stacked = _update_stack(obs)

    # Update tracking state
    _STATE["prev_obs"]    = obs.copy()
    _STATE["prev_stuck"]  = int(obs[17])
    _STATE["step"]       += 1
    _STATE["prev_attach"] = bool(np.any(obs[:16]))

    # Greedy action selection
    x = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(q_values))]
