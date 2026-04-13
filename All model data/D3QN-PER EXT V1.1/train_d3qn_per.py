"""
D3QN-PER Training Script for OBELIX Environment
================================================
D3QN = Dueling Double DQN
PER  = Prioritized Experience Replay with Sum-Tree

Improvements over vanilla DQN:
  1. DDQN       — decoupled action selection/evaluation, fixes overestimation
  2. Dueling     — separate V(s) and A(s,a) streams, faster learning in find phase
  3. PER         — prioritized replay by TD-error, fixes sparse reward sampling
  4. Soft update — TAU-weighted target sync, eliminates loss explosion
  5. Reward norm — stored/50, logged raw, fixes scale mismatch

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

Stuck/attach detection uses reward thresholds (no env variable access):
  -220 < reward < -100  →  stuck event
   95  < reward < 200   →  attachment event

Usage:
  python train_d3qn_per.py                          # default: no walls, difficulty 0
  python train_d3qn_per.py --wall_obstacles         # with walls
  python train_d3qn_per.py --resume weights_d3qn.pth
  python train_d3qn_per.py --episodes 5000 --wall_obstacles --difficulty 2
"""

import argparse
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX

# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
ACTIONS     = ("L45", "L22", "FW", "R22", "R45")
NUM_ACTIONS = len(ACTIONS)
ARENA_SIZE  = 500

OBS_DIM     = 18             # raw sensor observation dimension (unchanged)
EXT_OBS_DIM = 29             # extended observation with positional features
STACK_K     = 4
INPUT_DIM   = EXT_OBS_DIM * STACK_K   # 116

# Core DRL
GAMMA            = 0.99
LR               = 1e-4
BATCH_SIZE       = 256
REPLAY_CAPACITY  = 600_000
TRAIN_START      = 5_000
TRAIN_FREQ       = 8
TAU              = 0.005       # soft target update coefficient

# Exploration
EPS_START        = 0.4
EPS_END          = 0.01
EPS_DECAY_STEPS  = 2_000_000

# PER
PER_ALPHA        = 0.6
PER_BETA_START   = 0.4
PER_BETA_END     = 1.0
PER_BETA_STEPS   = 2_000_000
PER_EPS          = 1e-6

# Reward normalization
REWARD_SCALE     = 50.0

# Reward-based event detection thresholds
STUCK_REWARD_LO   = -220.0
STUCK_REWARD_HI   = -100.0
ATTACH_REWARD_LO  =   95.0
ATTACH_REWARD_HI  =  200.0

# Training and Logging
SAVE_PATH        = "weights_d3qn.pth"
LOG_EVERY        = 25
DEFAULT_EPISODES = 3_000
MAX_STEPS_PER_EP = 1_000
PATIENCE         = 5000
MIN_EPISODES_BEFORE_STOP = 2000
EMA_ALPHA        = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# Dead Reckoning Tracker
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
        self.local_heading = 0.0   # degrees, starts at 0 regardless of env init

    def update(self, action: str, stuck: bool) -> None:
        """
        Update tracker state given the action taken and whether the robot got stuck.

        Parameters
        ----------
        action : str   — one of the 5 ACTIONS
        stuck  : bool  — True if this step's reward indicated a stuck event
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
# Boundary Registry
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

    Distance features normalised by ARENA_SIZE; -1 sentinel for undetected.
    Zone flag: 0 or 1 based on which side of the wall line the robot is on;
               -1 if wall unknown.
    """

    STUCK_CAP       = 12
    ANGLE_THRESH_DEG = 2.0
    DIST_EPS        = 15.0    # pixels — absorbs up to one step of approach noise
    MIN_SEG_LEN     = 10.0    # pixels — minimum segment length for collinearity check

    def __init__(self):
        self.stuck_boundary_pts: list = []
        self.stuck_wall_pts:     list = []
        self.boundary_equations: list = [None, None, None, None]
        self.wall_equation              = None
        self._all_found                 = False

    # ── Line fitting ──────────────────────────────────────────────────────────
    def _fit_line(self, pts: list):
        """
        Fit ax + by + c = 0 to a list of (x,y) points using SVD.
        Returns normalised (a, b, c) with a²+b²=1, or None if degenerate.
        """
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

    # ── Distance ──────────────────────────────────────────────────────────────
    def _dist(self, pt: tuple, eq: tuple) -> float:
        """Signed-distance magnitude from point to normalised line."""
        a, b, c = eq
        x, y    = pt
        return abs(a * x + b * y + c)

    # ── Collinearity ──────────────────────────────────────────────────────────
    def _find_collinear_triplet(self, pts: list):
        """
        Check all C(n,3) triplets for collinearity.
        Returns list of 3 indices [i,j,k] or None if none found.
        """
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
                    sin_theta = min(sin_theta, 1.0)    # clamp for float safety
                    theta_deg = np.degrees(np.arcsin(sin_theta))
                    if theta_deg < self.ANGLE_THRESH_DEG:
                        return [i, j, k]
        return None

    # ── Core registration ─────────────────────────────────────────────────────
    def _try_fit_and_store(self, pts: list, is_wall: bool) -> list:
        """
        Attempt to find a collinear triplet in pts, fit a line, store it,
        and return the pruned pts list (triplet + pts on new line removed).

        Returns the (possibly unchanged) pts list.
        """
        if len(pts) < 3:
            return pts

        indices = self._find_collinear_triplet(pts)
        if indices is None:
            return pts

        triplet_pts = [pts[i] for i in indices]
        new_eq      = self._fit_line(triplet_pts)
        if new_eq is None:
            return pts    # degenerate fit, discard gracefully

        # Store equation in the appropriate slot
        if is_wall:
            self.wall_equation = new_eq
        else:
            for slot in range(4):
                if self.boundary_equations[slot] is None:
                    self.boundary_equations[slot] = new_eq
                    break

        # Remove triplet AND any other pts that lie on the new line
        indices_set = set(indices)
        pruned = [
            p for i, p in enumerate(pts)
            if i not in indices_set and self._dist(p, new_eq) >= self.DIST_EPS
        ]
        return pruned

    def register_stuck(self, pt: tuple, obs: np.ndarray, attach_flag: bool) -> None:
        """
        Process a stuck event.

        Parameters
        ----------
        pt          : (dx, dy) at the moment of the stuck event
        obs         : raw 18-dim observation at that step
        attach_flag : True if box attachment has already been detected
        """
        if self._all_found:
            return   # all equations known; no more discovery needed

        # ── Classify as wall or boundary ──────────────────────────────────────
        near_fwd    = bool(obs[4] or obs[6] or obs[8] or obs[10])
        ir_or_sonar = bool(obs[16]) or near_fwd
        is_wall     = (not attach_flag) and ir_or_sonar

        if is_wall:
            # Test against known wall equation first
            if self.wall_equation is not None:
                return   # already have wall line; discard

            # Guard against misclassification: if close to a known boundary, skip
            for eq in self.boundary_equations:
                if eq is not None and self._dist(pt, eq) < self.DIST_EPS:
                    return

            if len(self.stuck_wall_pts) < self.STUCK_CAP:
                self.stuck_wall_pts.append(pt)
            self.stuck_wall_pts = self._try_fit_and_store(self.stuck_wall_pts, is_wall=True)

        else:
            # Test against all known boundary equations
            for eq in self.boundary_equations:
                if eq is not None and self._dist(pt, eq) < self.DIST_EPS:
                    return   # already on a known boundary; discard

            # Guard against misclassification: if close to known wall, skip
            if self.wall_equation is not None and self._dist(pt, self.wall_equation) < self.DIST_EPS:
                return

            if len(self.stuck_boundary_pts) < self.STUCK_CAP:
                self.stuck_boundary_pts.append(pt)
            self.stuck_boundary_pts = self._try_fit_and_store(self.stuck_boundary_pts, is_wall=False)

        # Check if discovery is complete
        self._all_found = (
            all(eq is not None for eq in self.boundary_equations)
            and self.wall_equation is not None
        )

    def get_features(self, dx: float, dy: float) -> np.ndarray:
        """
        Compute 6 distance/zone features from current position.
        All distances normalised by ARENA_SIZE; -1 sentinel if undetected.

        Returns shape (6,) float32:
          [dist_b0, dist_b1, dist_b2, dist_b3, dist_wall, zone_flag]
        """
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
# Extended Observation Builder
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
      [0:18]  raw obs
      [18:22] geo features (dx, dy, cos, sin)
      [22]    attach_flag
      [23:29] registry distance features
    """
    geo   = tracker.get_geo_features()
    att   = np.array([float(attach_flag)], dtype=np.float32)
    dist  = registry.get_features(tracker.dx, tracker.dy)
    return np.concatenate([raw_obs.astype(np.float32), geo, att, dist])


# ══════════════════════════════════════════════════════════════════════════════
# Sum-Tree for O(log n) priority sampling
# ══════════════════════════════════════════════════════════════════════════════
class SumTree:
    """
    Binary sum tree for efficient priority-based sampling.

    Leaf nodes store transition priorities.
    Internal nodes store sums of their children.
    Sampling is O(log n), update is O(log n).

    Layout (capacity=4):
        Internal:  [0]
                  [1] [2]
        Leaves:  [3][4][5][6]   ← transitions stored here
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity + 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write_ptr  = 0
        self.n_entries  = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        leaf_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(leaf_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, value: float):
        """Sample a transition by value in [0, total]."""
        leaf_idx = self._retrieve(0, value)
        leaf_idx = int(np.clip(leaf_idx, self.capacity - 1, 2 * self.capacity - 1))
        data_idx = int(np.clip(leaf_idx - self.capacity + 1, 0, self.capacity - 1))
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


# ══════════════════════════════════════════════════════════════════════════════
# Prioritized Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════
class PrioritizedReplayBuffer:
    """
    Replay buffer using a sum-tree for O(log n) priority operations.

    On push: new transitions get max existing priority (optimistic init).
    On sample: proportional priority sampling with IS weight correction.
    On update: TD errors fed back after each gradient step.
    """

    def __init__(self, capacity: int, alpha: float):
        self.tree          = SumTree(capacity)
        self.alpha         = alpha
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done) -> None:
        transition = (
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int, beta: float):
        """
        Returns:
            batch       — (states, actions, rewards, next_states, dones)
            leaf_idxs   — tree indices needed for priority update
            is_weights  — importance sampling weights (normalised)
        """
        leaf_idxs  = []
        is_weights = []
        batch      = []

        segment = self.tree.total / batch_size
        min_prob = (
            np.min(self.tree.tree[self.tree.capacity - 1:
                                  self.tree.capacity - 1 + self.tree.n_entries])
            / self.tree.total
        ) if self.tree.n_entries > 0 else 1e-8
        if min_prob <= 0:
            min_prob = 1e-8
        max_weight = (min_prob * self.tree.n_entries) ** (-beta)

        for i in range(batch_size):
            transition = None
            attempts   = 0
            while transition is None and attempts < 10:
                lo       = segment * i
                hi       = segment * (i + 1)
                value    = random.uniform(lo, hi)
                leaf_idx, priority, transition = self.tree.get(value)
                attempts += 1

            if transition is None:
                for j in range(self.tree.n_entries):
                    if self.tree.data[j] is not None:
                        transition = self.tree.data[j]
                        leaf_idx   = j + self.tree.capacity - 1
                        priority   = self.tree.tree[leaf_idx]
                        break

            if transition is None:
                continue

            prob = priority / self.tree.total
            if prob <= 0:
                prob = 1e-8

            weight = ((prob * self.tree.n_entries) ** (-beta)) / max_weight
            is_weights.append(weight)
            leaf_idxs.append(leaf_idx)
            batch.append(transition)

        if len(batch) == 0:
            return None

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,                dtype=torch.long),
            torch.tensor(rewards,                dtype=torch.float32),
            torch.tensor(np.array(next_states),  dtype=torch.float32),
            torch.tensor(dones,                  dtype=torch.float32),
            leaf_idxs,
            torch.tensor(is_weights,             dtype=torch.float32),
        )

    def update_priorities(self, leaf_idxs: list, td_errors: np.ndarray) -> None:
        for idx, err in zip(leaf_idxs, td_errors):
            priority = (abs(float(err)) + PER_EPS) ** self.alpha
            self.tree.update(idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return len(self.tree)


# ══════════════════════════════════════════════════════════════════════════════
# Dueling DQN Network
# ══════════════════════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    """
    Dueling architecture splits Q(s,a) into V(s) + A(s,a).

    V(s)   — scalar value of being in state s (shared for all actions)
    A(s,a) — advantage of each action over the mean (zero-sum by construction)

    Q(s,a) = V(s) + [A(s,a) - mean_a(A(s,a))]

    Input widened to 116 (29-dim extended obs × stack 4).
    First shared layer widened to 384 to accommodate the richer,
    heterogeneous input (binary sensor bits + continuous positional features).
    """

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
# Frame Stack
# ══════════════════════════════════════════════════════════════════════════════
class FrameStack:
    def __init__(self, k: int, obs_dim: int):
        self.k       = k
        self.obs_dim = obs_dim
        self.frames: deque = deque(maxlen=k)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        for _ in range(self.k):
            self.frames.append(obs.copy())
        return self._get()

    def step(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.copy())
        return self._get()

    def _get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Schedules
# ══════════════════════════════════════════════════════════════════════════════
def get_epsilon(step: int) -> float:
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


def get_beta(step: int) -> float:
    frac = min(1.0, step / PER_BETA_STEPS)
    return PER_BETA_START + frac * (PER_BETA_END - PER_BETA_START)


# ══════════════════════════════════════════════════════════════════════════════
# Soft target update
# ══════════════════════════════════════════════════════════════════════════════
def soft_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.copy_(tau * p_online.data + (1.0 - tau) * p_target.data)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Networks ──────────────────────────────────────────────────────────────
    online_net = DuelingDQN(INPUT_DIM, NUM_ACTIONS).to(device)
    target_net = DuelingDQN(INPUT_DIM, NUM_ACTIONS).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    if args.resume and os.path.exists(args.resume):
        online_net.load_state_dict(torch.load(args.resume, map_location=device))
        target_net.load_state_dict(online_net.state_dict())
        print(f"Resumed from {args.resume}")

    optimizer = optim.Adam(online_net.parameters(), lr=LR)

    # ── Environment ───────────────────────────────────────────────────────────
    env = OBELIX(
        scaling_factor=5,
        arena_size=ARENA_SIZE,
        max_steps=MAX_STEPS_PER_EP,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=0,
    )

    replay = PrioritizedReplayBuffer(REPLAY_CAPACITY, PER_ALPHA)
    fstack = FrameStack(STACK_K, EXT_OBS_DIM)

    # ── Tracking ──────────────────────────────────────────────────────────────
    total_steps       = 0
    episode_rewards   = []
    episode_successes = []
    losses            = []
    stuck_counts      = []
    best_mean         = -np.inf
    ema_reward        = None
    no_improve        = 0

    wall_str = "walls=ON" if args.wall_obstacles else "walls=OFF"
    print(
        f"\nTraining D3QN-PER | episodes={args.episodes} | {wall_str} | "
        f"difficulty={args.difficulty}"
    )
    print("=" * 70)

    rng_master = np.random.default_rng(42)

    for ep in range(1, args.episodes + 1):

        # ── Episode initialisation ────────────────────────────────────────────
        seed    = int(rng_master.integers(10_000, 10_000_000))
        raw_obs = env.reset(seed=seed)

        tracker     = DeadReckoningTracker()
        registry    = BoundaryRegistry()
        attach_flag = False

        ext_obs = build_extended_obs(raw_obs, tracker, attach_flag, registry)
        state   = fstack.reset(ext_obs)

        ep_reward     = 0.0
        ep_stuck_count = 0
        done           = False

        while not done:
            # ε-greedy action selection
            eps = get_epsilon(total_steps)
            if random.random() < eps:
                action_idx = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    s_t        = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    action_idx = int(online_net(s_t).argmax(dim=1).item())

            action_str          = ACTIONS[action_idx]
            raw_next, reward, done = env.step(action_str, render=args.render)

            # ── Event detection from reward ────────────────────────────────────
            step_stuck  = STUCK_REWARD_LO  < reward < STUCK_REWARD_HI
            step_attach = ATTACH_REWARD_LO < reward < ATTACH_REWARD_HI

            if step_attach:
                attach_flag = True

            if step_stuck:
                ep_stuck_count += 1

            # ── Update dead reckoning tracker ──────────────────────────────────
            tracker.update(action_str, step_stuck)

            # ── Register stuck event with boundary registry ────────────────────
            if action_str == "FW" and step_stuck:
                registry.register_stuck(
                    tracker.get_position(), raw_next, attach_flag
                )

            # ── Build extended next observation ────────────────────────────────
            ext_next   = build_extended_obs(raw_next, tracker, attach_flag, registry)
            next_state = fstack.step(ext_next)

            # ── Reward shaping ─────────────────────────────────────────────────
            wall_hit = step_stuck and not attach_flag
            if wall_hit:
                extra  = -50 if args.wall_obstacles else -300
                shaped = reward + extra
            else:
                shaped = reward

            stored_reward = shaped / REWARD_SCALE

            last_data_idx = replay.tree.write_ptr
            replay.push(state, action_idx, stored_reward, next_state, done)

            state      = next_state
            ep_reward += reward
            total_steps += 1

            # ── Gradient update ────────────────────────────────────────────────
            if len(replay) >= TRAIN_START and total_steps % TRAIN_FREQ == 0:
                beta   = get_beta(total_steps)
                result = replay.sample(BATCH_SIZE, beta)
                if result is None:
                    continue
                (
                    s, a, r, s2, d,
                    leaf_idxs, is_weights
                ) = replay.sample(BATCH_SIZE, beta)

                s, a, r, s2, d = (
                    s.to(device), a.to(device), r.to(device),
                    s2.to(device), d.to(device),
                )
                is_weights = is_weights.to(device)

                with torch.no_grad():
                    best_actions = online_net(s2).argmax(dim=1, keepdim=True)
                    q_next       = target_net(s2).gather(1, best_actions).squeeze(1)
                    q_target     = r + GAMMA * q_next * (1.0 - d)

                q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                td_errors = (q_pred - q_target).detach().cpu().numpy()
                replay.update_priorities(leaf_idxs, td_errors)

                element_loss = nn.functional.smooth_l1_loss(
                    q_pred, q_target, reduction="none"
                )
                loss = (is_weights * element_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                soft_update(online_net, target_net, TAU)
                losses.append(loss.item())

        # ── Ground-truth success detection ────────────────────────────────────
        success = (
            env.enable_push
            and env._box_touches_boundary(env.box_center_x, env.box_center_y)
        )
        if success:
            steps_remaining  = env.max_steps - env.current_step
            efficiency_bonus = (steps_remaining / env.max_steps) * 500 / REWARD_SCALE
            data_idx         = int(np.clip(last_data_idx, 0, replay.tree.capacity - 1))
            if replay.tree.data[data_idx] is not None:
                s, a, r, s2, d = replay.tree.data[data_idx]
                replay.tree.data[data_idx] = (s, a, r + efficiency_bonus, s2, d)

        episode_rewards.append(ep_reward)
        episode_successes.append(success)
        stuck_counts.append(ep_stuck_count)

        # ── Periodic priority refresh ──────────────────────────────────────────
        if ep % 100 == 0:
            n             = replay.tree.n_entries
            refresh_count = n // 50
            oldest_start  = replay.tree.write_ptr % replay.tree.capacity
            for i in range(refresh_count):
                idx = (oldest_start + i) % replay.tree.capacity
                if replay.tree.data[idx] is None:
                    continue
                leaf_idx = int(np.clip(
                    idx + replay.tree.capacity - 1,
                    replay.tree.capacity - 1,
                    2 * replay.tree.capacity - 1
                ))
                replay.tree.update(leaf_idx, replay._max_priority)

        # ── Logging ───────────────────────────────────────────────────────────
        if ep % LOG_EVERY == 0:
            window     = min(LOG_EVERY, len(episode_rewards))
            mean_r     = np.mean(episode_rewards[-window:])
            mean_s     = np.mean(episode_successes[-window:]) * 100
            mean_l     = np.mean(losses[-200:]) if losses else 0.0
            eps_now    = get_epsilon(total_steps)
            beta_now   = get_beta(total_steps)
            mean_stuck = np.mean(stuck_counts[-window:])

            print(
                f"Ep {ep:5d}/{args.episodes} | "
                f"steps={total_steps:8d} | "
                f"mean_r={mean_r:9.1f} | "
                f"success={mean_s:5.1f}% | "
                f"loss={mean_l:.4f} | "
                f"eps={eps_now:.3f} | "
                f"beta={beta_now:.3f} | "
                f"stuck={mean_stuck:6.1f}"
            )

            if ema_reward is None:
                ema_reward = mean_r
            else:
                ema_reward = EMA_ALPHA * mean_r + (1 - EMA_ALPHA) * ema_reward

            if ema_reward > best_mean:
                best_mean  = ema_reward
                no_improve = 0
                torch.save(online_net.state_dict(), SAVE_PATH)
                print(f"  ✓ New best (ema_r={best_mean:.1f}) → {SAVE_PATH}")
            else:
                no_improve += LOG_EVERY
                if no_improve >= PATIENCE and ep >= MIN_EPISODES_BEFORE_STOP:
                    print(
                        f"Early stopping at episode {ep} — "
                        f"no improvement in {PATIENCE} episodes"
                    )
                    break

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = "weights_d3qn_final.pth"
    torch.save(online_net.state_dict(), final_path)
    print(f"\nTraining complete.")
    print(f"  Best model  (ema_r={best_mean:.1f}) → {SAVE_PATH}")
    print(f"  Final model → {final_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int,  default=DEFAULT_EPISODES)
    parser.add_argument("--render",         action="store_true")
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="0=static box, 2=blinking, 3=moving+blinking",
    )
    parser.add_argument(
        "--box_speed",
        type=int,
        default=2,
        help="box speed in pixels/step (difficulty>=3 only)",
    )
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)
