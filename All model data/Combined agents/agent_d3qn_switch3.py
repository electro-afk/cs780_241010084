"""
D3QN Agent — Model Switching After 3 Wall Encounters
=====================================================
Starts every episode with the no-walls model.
Switches to the walls model only after 3 confirmed wall stuck events.
Switch does NOT persist across episodes — resets at every episode boundary.

Using 3 events instead of 1 reduces the risk of a false positive switch
caused by forward sonars firing on the box during a stuck event before
the attach_flag has latched.

Wall detection:
  ir_or_sonar = obs[16] OR obs[4] OR obs[6] OR obs[8] OR obs[10]
  is_wall_hit = ir_or_sonar AND obs[17]==1 AND NOT attach_flag

Place both weight files in the same directory as this script:
  weights_no_walls.pth   ← Phase 1 best model
  weights_walls.pth      ← Phase 2 best model

Evaluate locally:
  python evaluate.py --agent_file agent_d3qn_switch3.py --difficulty 0 --wall_obstacles
"""

import os
from collections import deque
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
NUM_ACTIONS  = len(ACTIONS)
ARENA_SIZE   = 500

OBS_DIM      = 18
EXT_OBS_DIM  = 29
STACK_K      = 4
INPUT_DIM    = EXT_OBS_DIM * STACK_K   # 116

WEIGHTS_NO_WALLS = "weights_d3qn_V1.pth"
WEIGHTS_WALLS    = "weights_d3qn_V3.pth"

# Switch threshold — number of confirmed wall events before switching
WALL_SWITCH_THRESHOLD = 3


# ══════════════════════════════════════════════════════════════════════════════
# Dead Reckoning Tracker
# ══════════════════════════════════════════════════════════════════════════════
class DeadReckoningTracker:
    ANGLE_DELTAS = {
        "L45":  45.0, "L22":  22.5,
        "FW":    0.0,
        "R22": -22.5, "R45": -45.0,
    }
    FORWARD_STEP = 5.0

    def __init__(self):
        self.dx            = 0.0
        self.dy            = 0.0
        self.local_heading = 0.0

    def update(self, action: str, stuck: bool) -> None:
        self.local_heading += self.ANGLE_DELTAS[action]
        if action == "FW" and not stuck:
            rad = np.deg2rad(self.local_heading)
            self.dx += self.FORWARD_STEP * np.cos(rad)
            self.dy += self.FORWARD_STEP * np.sin(rad)

    def get_position(self):
        return (self.dx, self.dy)

    def get_geo_features(self) -> np.ndarray:
        rad = np.deg2rad(self.local_heading)
        return np.array([
            self.dx / ARENA_SIZE,
            self.dy / ARENA_SIZE,
            np.cos(rad),
            np.sin(rad),
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Boundary Registry
# ══════════════════════════════════════════════════════════════════════════════
class BoundaryRegistry:
    STUCK_CAP        = 12
    ANGLE_THRESH_DEG = 2.0
    DIST_EPS         = 15.0
    MIN_SEG_LEN      = 10.0

    def __init__(self):
        self.stuck_boundary_pts: list = []
        self.stuck_wall_pts:     list = []
        self.boundary_equations: list = [None, None, None, None]
        self.wall_equation              = None
        self._all_found                 = False
        self.wall_hit_count             = 0    # tracks wall events this episode

    def _fit_line(self, pts: list):
        arr = np.array(pts, dtype=np.float64)
        A   = np.column_stack([arr, np.ones(len(arr))])
        try:
            _, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return None
        a, b, c = Vt[-1]
        norm = np.sqrt(a * a + b * b)
        if norm < 1e-10:
            return None
        return (a / norm, b / norm, c / norm)

    def _dist(self, pt: tuple, eq: tuple) -> float:
        a, b, c = eq
        x, y    = pt
        return abs(a * x + b * y + c)

    def _find_collinear_triplet(self, pts: list):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    A  = np.array(pts[i], dtype=np.float64)
                    B  = np.array(pts[j], dtype=np.float64)
                    C  = np.array(pts[k], dtype=np.float64)
                    AB = B - A
                    AC = C - A
                    len_AB = np.linalg.norm(AB)
                    len_AC = np.linalg.norm(AC)
                    if len_AB < self.MIN_SEG_LEN or len_AC < self.MIN_SEG_LEN:
                        continue
                    cross     = AB[0] * AC[1] - AB[1] * AC[0]
                    sin_theta = min(abs(cross) / (len_AB * len_AC), 1.0)
                    if np.degrees(np.arcsin(sin_theta)) < self.ANGLE_THRESH_DEG:
                        return [i, j, k]
        return None

    def _try_fit_and_store(self, pts: list, is_wall: bool) -> list:
        if len(pts) < 3:
            return pts
        indices = self._find_collinear_triplet(pts)
        if indices is None:
            return pts
        new_eq = self._fit_line([pts[i] for i in indices])
        if new_eq is None:
            return pts
        if is_wall:
            self.wall_equation = new_eq
        else:
            for slot in range(4):
                if self.boundary_equations[slot] is None:
                    self.boundary_equations[slot] = new_eq
                    break
        idx_set = set(indices)
        return [
            p for i, p in enumerate(pts)
            if i not in idx_set and self._dist(p, new_eq) >= self.DIST_EPS
        ]

    def register_stuck(self, pt: tuple, obs: np.ndarray, attach_flag: bool) -> None:
        if self._all_found:
            return

        near_fwd    = bool(obs[4] or obs[6] or obs[8] or obs[10])
        ir_or_sonar = bool(obs[16]) or near_fwd
        is_wall     = (not attach_flag) and ir_or_sonar

        if is_wall:
            self.wall_hit_count += 1   # increment regardless of equation status
            if self.wall_equation is not None:
                return
            for eq in self.boundary_equations:
                if eq is not None and self._dist(pt, eq) < self.DIST_EPS:
                    return
            if len(self.stuck_wall_pts) < self.STUCK_CAP:
                self.stuck_wall_pts.append(pt)
            self.stuck_wall_pts = self._try_fit_and_store(self.stuck_wall_pts, is_wall=True)
        else:
            for eq in self.boundary_equations:
                if eq is not None and self._dist(pt, eq) < self.DIST_EPS:
                    return
            if self.wall_equation is not None and self._dist(pt, self.wall_equation) < self.DIST_EPS:
                return
            if len(self.stuck_boundary_pts) < self.STUCK_CAP:
                self.stuck_boundary_pts.append(pt)
            self.stuck_boundary_pts = self._try_fit_and_store(
                self.stuck_boundary_pts, is_wall=False
            )

        self._all_found = (
            all(eq is not None for eq in self.boundary_equations)
            and self.wall_equation is not None
        )

    def get_features(self, dx: float, dy: float) -> np.ndarray:
        pt       = (dx, dy)
        features = []
        for eq in self.boundary_equations:
            if eq is None:
                features.append(-1.0)
            else:
                features.append(self._dist(pt, eq) / ARENA_SIZE)
        if self.wall_equation is None:
            features.append(-1.0)
            features.append(-1.0)
        else:
            features.append(self._dist(pt, self.wall_equation) / ARENA_SIZE)
            a, b, c = self.wall_equation
            features.append(0.0 if (a * dx + b * dy + c) < 0.0 else 1.0)
        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Extended Observation Builder
# ══════════════════════════════════════════════════════════════════════════════
def build_extended_obs(
    raw_obs:     np.ndarray,
    tracker:     DeadReckoningTracker,
    attach_flag: bool,
    registry:    BoundaryRegistry,
) -> np.ndarray:
    geo  = tracker.get_geo_features()
    att  = np.array([float(attach_flag)], dtype=np.float32)
    dist = registry.get_features(tracker.dx, tracker.dy)
    return np.concatenate([raw_obs.astype(np.float32), geo, att, dist])


# ══════════════════════════════════════════════════════════════════════════════
# Dueling DQN Network
# ══════════════════════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 384), nn.ReLU(),
            nn.Linear(384, 256),       nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        return V + (A - A.mean(dim=1, keepdim=True))


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════
def _load_model(filename: str) -> DuelingDQN:
    path  = os.path.join(os.path.dirname(__file__), filename)
    model = DuelingDQN(INPUT_DIM, NUM_ACTIONS)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "weights" in ckpt:
            model.load_state_dict(ckpt["weights"])
        else:
            model.load_state_dict(ckpt)
        print(f"[agent] Loaded {filename}")
    else:
        print(f"[agent] WARNING: {filename} not found. Using random weights.")
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Global agent state
# ══════════════════════════════════════════════════════════════════════════════
_STATE: dict = {
    # Models — loaded once, persist forever
    "model_no_walls": None,
    "model_walls":    None,
    # Active model — reset each episode
    "active_model":   None,
    # Frame stack
    "frames":         None,
    # Episode reset detection
    "prev_obs":       None,
    "was_done":       False,
    "prev_stuck":     0,
    "prev_attach":    False,
    # Per-episode tracking
    "step":           0,
    "tracker":        None,
    "registry":       None,
    "attach_flag":    False,
    "prev_action":    None,
    "wall_switched":  False,   # has the switch already happened this episode
}


def _ensure_models_loaded():
    if _STATE["model_no_walls"] is None:
        _STATE["model_no_walls"] = _load_model(WEIGHTS_NO_WALLS)
    if _STATE["model_walls"] is None:
        _STATE["model_walls"]    = _load_model(WEIGHTS_WALLS)


# ══════════════════════════════════════════════════════════════════════════════
# Episode reset detection
# ══════════════════════════════════════════════════════════════════════════════
def _is_new_episode(obs: np.ndarray) -> bool:
    if _STATE["prev_obs"] is None:
        return True
    if _STATE["was_done"]:
        return True
    # Post-success: attachment was active, now everything silent
    prev       = _STATE["prev_obs"]
    all_silent = not np.any(obs[:17])
    if _STATE["prev_attach"] and all_silent:
        return True
    return False


def _init_episode(obs: np.ndarray) -> np.ndarray:
    _ensure_models_loaded()

    tracker     = DeadReckoningTracker()
    registry    = BoundaryRegistry()
    attach_flag = False

    _STATE["tracker"]       = tracker
    _STATE["registry"]      = registry
    _STATE["attach_flag"]   = attach_flag
    _STATE["prev_action"]   = None
    _STATE["step"]          = 0
    _STATE["was_done"]      = False
    _STATE["prev_attach"]   = False
    _STATE["wall_switched"] = False

    # Always start with the no-walls model
    _STATE["active_model"]  = _STATE["model_no_walls"]

    ext    = build_extended_obs(obs, tracker, attach_flag, registry)
    frames = deque(maxlen=STACK_K)
    for _ in range(STACK_K):
        frames.append(ext.copy())
    _STATE["frames"] = frames

    return np.concatenate(list(frames))


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Choose an action from the current 18-bit observation.
    Switches from no-walls model to walls model after 3 confirmed wall stuck
    events. Switch resets at every episode boundary.

    Using 3 events rather than 1 reduces false positives from forward sonars
    firing on the box before the attach_flag has had a chance to latch.
    """
    if _is_new_episode(obs):
        stacked = _init_episode(obs)
    else:
        tracker     = _STATE["tracker"]
        registry    = _STATE["registry"]
        prev_action = _STATE["prev_action"]

        if prev_action is not None:
            stuck = bool(obs[17] == 1)
            tracker.update(prev_action, stuck)

            if prev_action == "FW" and stuck:
                registry.register_stuck(
                    tracker.get_position(), obs, _STATE["attach_flag"]
                )

        # ── Attach flag ───────────────────────────────────────────────────────
        near_fwd    = bool(obs[4] or obs[6] or obs[8] or obs[10])
        ir_or_sonar = bool(obs[16]) or near_fwd
        if ir_or_sonar and obs[17] == 0:
            _STATE["attach_flag"] = True

        # ── Model switch check ────────────────────────────────────────────────
        # Switch only after WALL_SWITCH_THRESHOLD=3 confirmed wall events.
        # wall_hit_count increments inside register_stuck whenever is_wall=True,
        # giving us a more robust signal than a single potentially noisy event.
        if (not _STATE["wall_switched"] and
                registry.wall_hit_count >= WALL_SWITCH_THRESHOLD):
            _STATE["active_model"]  = _STATE["model_walls"]
            _STATE["wall_switched"] = True
            print(f"[agent_switch3] {WALL_SWITCH_THRESHOLD} wall events confirmed "
                  f"at step {_STATE['step']} — switching to walls model")

        ext = build_extended_obs(obs, tracker, _STATE["attach_flag"], registry)
        _STATE["frames"].append(ext.copy())
        stacked = np.concatenate(list(_STATE["frames"]))

    # ── Update persistent tracking state ─────────────────────────────────────
    _STATE["prev_obs"]    = obs.copy()
    _STATE["prev_stuck"]  = int(obs[17])
    _STATE["step"]       += 1
    _STATE["prev_attach"] = bool(np.any(obs[:16]))

    # ── Greedy inference using active model ───────────────────────────────────
    x = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = _STATE["active_model"](x).squeeze(0).numpy()

    action = ACTIONS[int(np.argmax(q_values))]
    _STATE["prev_action"] = action
    return action
