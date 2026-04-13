"""
D3QN Agent — Submission file for OBELIX RL Challenge
=====================================================
Trained with: D3QN-PER (Dueling Double DQN + Prioritized Experience Replay)

PER infrastructure is training-only — completely absent here.

Extended observation (29-dim per frame, 116-dim stacked):
  [0:18]  raw sensor bits (unchanged)
  [18]    dx / arena_size         — dead-reckoning x displacement
  [19]    dy / arena_size         — dead-reckoning y displacement
  [20]    cos(local_heading)      — heading cosine, avoids wrap
  [21]    sin(local_heading)      — heading sine
  [22]    attach_flag             — 1 after first box attachment
  [23:27] dist to boundary_eq 0..3 / arena_size, -1 if undetected
  [27]    dist to wall_eq / arena_size, -1 if undetected
  [28]    zone_flag               — 0/1 side of wall, -1 if wall unknown

Attach detection (observation-only, no reward access):
  attach_flag latches True when:
    (obs[16] OR obs[4] OR obs[6] OR obs[8] OR obs[10]) AND obs[17]==0
  i.e. IR/sonar contact with NO stuck flag — only possible during push phase.

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
NUM_ACTIONS  = len(ACTIONS)
ARENA_SIZE   = 500

OBS_DIM      = 18
EXT_OBS_DIM  = 29
STACK_K      = 4
INPUT_DIM    = EXT_OBS_DIM * STACK_K   # 116

WEIGHTS_FILE = "weights_d3qn.pth"


# ══════════════════════════════════════════════════════════════════════════════
# Dead Reckoning Tracker  (identical to train_d3qn_per.py)
# ══════════════════════════════════════════════════════════════════════════════
class DeadReckoningTracker:
    """
    Tracks robot displacement (dx, dy) and local heading in the agent's own
    coordinate frame.

    The local frame is a pure translation of the world frame, offset by the
    unknown starting position, but rotated by the unknown initial heading.
    Consequently arena boundaries appear as diagonal lines in this frame —
    handled by the BoundaryRegistry collinearity approach.

    Heading convention matches the environment:
      L45 → +45°,  L22 → +22.5°,  R22 → −22.5°,  R45 → −45°

    Position update rules:
      - Any rotation action: update heading only, position unchanged
      - FW + not stuck:      update heading (no change) and position
      - FW + stuck:          update heading (no change), position unchanged
                             caller should register stuck event with BoundaryRegistry
    """

    ANGLE_DELTAS = {
        "L45": 45.0,
        "L22": 22.5,
        "FW":   0.0,
        "R22": -22.5,
        "R45": -45.0,
    }
    FORWARD_STEP = 5.0

    def __init__(self):
        self.dx            = 0.0
        self.dy            = 0.0
        self.local_heading = 0.0

    def update(self, action: str, stuck: bool) -> None:
        """
        Update tracker state given the action taken and whether the robot got stuck.

        Parameters
        ----------
        action : str   — one of the 5 ACTIONS
        stuck  : bool  — True if obs[17]==1 on this step
        """
        self.local_heading += self.ANGLE_DELTAS[action]

        if action == "FW" and not stuck:
            rad = np.deg2rad(self.local_heading)
            self.dx += self.FORWARD_STEP * np.cos(rad)
            self.dy += self.FORWARD_STEP * np.sin(rad)

    def get_position(self):
        """Return current (dx, dy) as a tuple."""
        return (self.dx, self.dy)

    def get_geo_features(self) -> np.ndarray:
        """
        Returns 4 normalised positional features:
          [dx/ARENA_SIZE, dy/ARENA_SIZE, cos(heading), sin(heading)]
        """
        rad = np.deg2rad(self.local_heading)
        return np.array([
            self.dx / ARENA_SIZE,
            self.dy / ARENA_SIZE,
            np.cos(rad),
            np.sin(rad),
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Boundary Registry  (identical to train_d3qn_per.py)
# ══════════════════════════════════════════════════════════════════════════════
class BoundaryRegistry:
    """
    Discovers and stores straight-line equations for the 4 arena boundaries
    and 1 wall obstacle in the robot's local dead-reckoning frame.

    Each line is stored as normalised (a, b, c) where a²+b²=1 and the
    signed distance from point (x,y) is |ax + by + c|.

    Discovery algorithm:
      1. On each stuck event, classify as wall or boundary via:
           ir_or_sonar = obs[16] OR obs[4] OR obs[6] OR obs[8] OR obs[10]
           is_wall     = ir_or_sonar AND NOT attach_flag
      2. Test point against all known equations; discard if within DIST_EPS.
      3. Otherwise append to the appropriate stuck array (cap: STUCK_CAP=12).
      4. Check all C(n,3) triplets for collinearity via arcsin of normalised
         cross product; threshold ANGLE_THRESH_DEG=2°; skip degenerate pairs
         where segment length < MIN_SEG_LEN=10px.
      5. On finding a collinear triplet, fit line via SVD, store equation,
         remove triplet plus any other stuck pts that lie on the new line.
      6. Once all 5 equations found, collinearity checks stop; new stuck pts
         are only tested against known equations and discarded.

    Distance features normalised by ARENA_SIZE; -1 sentinel if undetected.
    Zone flag: 0 or 1 based on which side of wall line the robot is on;
               -1 if wall unknown.
    """

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
                    sin_theta = abs(cross) / (len_AB * len_AC)
                    sin_theta = min(sin_theta, 1.0)
                    theta_deg = np.degrees(np.arcsin(sin_theta))
                    if theta_deg < self.ANGLE_THRESH_DEG:
                        return [i, j, k]
        return None

    def _try_fit_and_store(self, pts: list, is_wall: bool) -> list:
        if len(pts) < 3:
            return pts

        indices = self._find_collinear_triplet(pts)
        if indices is None:
            return pts

        triplet_pts = [pts[i] for i in indices]
        new_eq      = self._fit_line(triplet_pts)
        if new_eq is None:
            return pts

        if is_wall:
            self.wall_equation = new_eq
        else:
            for slot in range(4):
                if self.boundary_equations[slot] is None:
                    self.boundary_equations[slot] = new_eq
                    break

        indices_set = set(indices)
        pruned = [
            p for i, p in enumerate(pts)
            if i not in indices_set and self._dist(p, new_eq) >= self.DIST_EPS
        ]
        return pruned

    def register_stuck(self, pt: tuple, obs: np.ndarray, attach_flag: bool) -> None:
        if self._all_found:
            return

        near_fwd    = bool(obs[4] or obs[6] or obs[8] or obs[10])
        ir_or_sonar = bool(obs[16]) or near_fwd
        is_wall     = (not attach_flag) and ir_or_sonar

        if is_wall:
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
            self.stuck_boundary_pts = self._try_fit_and_store(self.stuck_boundary_pts, is_wall=False)

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
            zone    = 0.0 if (a * dx + b * dy + c) < 0.0 else 1.0
            features.append(zone)

        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Extended Observation Builder  (identical to train_d3qn_per.py)
# ══════════════════════════════════════════════════════════════════════════════
def build_extended_obs(
    raw_obs:     np.ndarray,
    tracker:     DeadReckoningTracker,
    attach_flag: bool,
    registry:    BoundaryRegistry,
) -> np.ndarray:
    """
    Concatenate raw sensor bits with positional and spatial features.
    Output shape: (29,)
    """
    geo  = tracker.get_geo_features()
    att  = np.array([float(attach_flag)], dtype=np.float32)
    dist = registry.get_features(tracker.dx, tracker.dy)
    return np.concatenate([raw_obs.astype(np.float32), geo, att, dist])


# ══════════════════════════════════════════════════════════════════════════════
# Dueling Network  (must be identical to train_d3qn_per.py)
# ══════════════════════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
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
    # Frame stack
    "frames":       None,
    # Episode reset detection
    "prev_obs":     None,
    "was_done":     False,
    "prev_stuck":   0,
    "prev_attach":  False,   # any sensor contact, for reset detection only
    # Step counter
    "step":         0,
    # New: positional and spatial tracking
    "tracker":      None,
    "registry":     None,
    "attach_flag":  False,   # latching push-phase flag
    "prev_action":  None,    # action taken at previous step, for tracker update
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

    prev       = _STATE["prev_obs"]
    all_silent = not np.any(obs[:17])
    had_signal = np.any(prev[:17])

    if _STATE["prev_attach"] and all_silent:
        return True

    if _STATE["prev_stuck"] > 0 and obs[17] == 0 and all_silent and had_signal:
        return True

    return False


def _init_episode(obs: np.ndarray) -> np.ndarray:
    """
    Reset all per-episode state and return the initial stacked observation.
    """
    tracker     = DeadReckoningTracker()
    registry    = BoundaryRegistry()
    attach_flag = False

    _STATE["tracker"]      = tracker
    _STATE["registry"]     = registry
    _STATE["attach_flag"]  = attach_flag
    _STATE["prev_action"]  = None
    _STATE["step"]         = 0
    _STATE["was_done"]     = False
    _STATE["prev_attach"]  = False

    ext = build_extended_obs(obs, tracker, attach_flag, registry)

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

    # ── Episode boundary handling ──────────────────────────────────────────────
    if _is_new_episode(obs):
        stacked = _init_episode(obs)
    else:
        tracker     = _STATE["tracker"]
        registry    = _STATE["registry"]
        prev_action = _STATE["prev_action"]

        # Update tracker and registry using the previous action and current obs
        if prev_action is not None:
            stuck = bool(obs[17] == 1)
            tracker.update(prev_action, stuck)

            # Register stuck event with boundary registry
            if prev_action == "FW" and stuck:
                registry.register_stuck(
                    tracker.get_position(), obs, _STATE["attach_flag"]
                )

        # Update attach flag:
        # IR or forward near sonars firing WITH no stuck = robot is freely
        # pushing the box → attachment confirmed. Latch permanently.
        near_fwd    = bool(obs[4] or obs[6] or obs[8] or obs[10])
        ir_or_sonar = bool(obs[16]) or near_fwd
        if ir_or_sonar and obs[17] == 0:
            _STATE["attach_flag"] = True

        # Build extended observation and push onto frame stack
        ext = build_extended_obs(obs, tracker, _STATE["attach_flag"], registry)
        _STATE["frames"].append(ext.copy())
        stacked = np.concatenate(list(_STATE["frames"]))

    # ── Update tracking state for next call ───────────────────────────────────
    _STATE["prev_obs"]    = obs.copy()
    _STATE["prev_stuck"]  = int(obs[17])
    _STATE["step"]       += 1
    _STATE["prev_attach"] = bool(np.any(obs[:16]))   # for reset detection only

    # ── Greedy action selection ────────────────────────────────────────────────
    x = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(x).squeeze(0).numpy()

    action = ACTIONS[int(np.argmax(q_values))]
    _STATE["prev_action"] = action
    return action
