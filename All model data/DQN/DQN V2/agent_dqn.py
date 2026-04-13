"""
DQN Agent — Submission file for OBELIX RL Challenge
=====================================================
Trained on: difficulty=0, wall_obstacles=True

This file is self-contained. It:
  1. Defines the same DQN network used during training.
  2. Loads weights from weights_dqn.pth (placed in the same directory).
  3. Maintains a frame-stack of k=4 observations in a global state dict
     so that the stateless policy() API can still use temporal context.
  4. Detects episode resets by monitoring the attachment bit (obs[17])
     and the running step counter, then flushes the frame buffer.

Evaluate locally:
  python evaluate.py --agent_file agent_dqn.py --wall_obstacles --difficulty 0
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
OBS_DIM = 18
STACK_K = 4
INPUT_DIM = OBS_DIM * STACK_K  # 72

# ══════════════════════════════════════════════════════════════════════════════
# Network (must match train_dqn.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
class DQN(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Global agent state  (persists across policy() calls within one episode)
# ══════════════════════════════════════════════════════════════════════════════
_STATE: dict = {
    "model":       None,           # loaded DQN
    "frames":      None,           # deque of k obs arrays
    "prev_obs":    None,           # previous raw observation
    "step":        0,              # step counter within episode
    "was_done":    False,          # True after episode terminated
    "prev_stuck":  0,              # previous stuck_flag (obs[17])
    "prev_attach": False,          # previous enable_push inference
}


def _load_model() -> DQN:
    """Load weights once; cached in _STATE['model']."""
    if _STATE["model"] is not None:
        return _STATE["model"]

    weights_path = os.path.join(os.path.dirname(__file__), "weights_dqn.pth")
    model = DQN(INPUT_DIM, NUM_ACTIONS)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        print(f"[agent_dqn] Loaded weights from {weights_path}")
    else:
        # No weights found — fall back to forward-biased random walk
        print(f"[agent_dqn] WARNING: {weights_path} not found. Using random policy.")

    _STATE["model"] = model
    return model


def _is_new_episode(obs: np.ndarray) -> bool:
    """
    Heuristic episode-reset detector.

    A new episode is detected when:
      - This is the very first call (_STATE['prev_obs'] is None), OR
      - The previous call was marked done (_STATE['was_done']), OR
      - The stuck flag (obs[17]) dropped from ≥1 to 0 AND all sensors went
        quiet simultaneously (strong signal of a teleport/reset), OR
      - Attachment was active last step but all sensor bits are now zero
        (only happens at reset after a successful push to boundary).
    """
    if _STATE["prev_obs"] is None:
        return True
    if _STATE["was_done"]:
        return True

    prev = _STATE["prev_obs"]

    # All sensor bits silent after previously having signal → likely reset
    all_silent_now = not np.any(obs[:17])
    had_signal = np.any(prev[:17])

    # Attachment was on, now everything is silent → episode ended + new start
    if _STATE["prev_attach"] and all_silent_now:
        return True

    # Stuck flag cleared AND sensors went from noisy to silent → teleport/reset
    if _STATE["prev_stuck"] > 0 and obs[17] == 0 and all_silent_now and had_signal:
        return True

    return False


def _init_episode(obs: np.ndarray) -> np.ndarray:
    """Flush frame stack and fill with the first observation."""
    _STATE["frames"] = deque(maxlen=STACK_K)
    for _ in range(STACK_K):
        _STATE["frames"].append(obs.copy())
    _STATE["step"] = 0
    _STATE["was_done"] = False
    _STATE["prev_attach"] = False
    return np.concatenate(list(_STATE["frames"]), axis=0)


def _update_stack(obs: np.ndarray) -> np.ndarray:
    """Push latest obs into the rolling frame stack."""
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
        RNG provided by the evaluator (used for fallback random action).

    Returns
    -------
    str  —  one of "L45", "L22", "FW", "R22", "R45"
    """
    model = _load_model()

    # ── Episode reset detection ───────────────────────────────────────────────
    if _is_new_episode(obs):
        stacked = _init_episode(obs)
    else:
        stacked = _update_stack(obs)

    # ── Update state tracking ─────────────────────────────────────────────────
    _STATE["prev_obs"] = obs.copy()
    _STATE["prev_stuck"] = int(obs[17])
    _STATE["step"] += 1

    # Infer attachment: bit 17 is stuck_flag; we track if ANY sensor just lit up
    # alongside forward sensors — crude but sufficient for reset detection.
    _STATE["prev_attach"] = bool(np.any(obs[:16]))  # any sonar active

    # ── Model inference ───────────────────────────────────────────────────────
    x = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(x).squeeze(0).numpy()

    action_idx = int(np.argmax(q_values))
    return ACTIONS[action_idx]
