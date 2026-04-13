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

Usage:
  python train_d3qn_per.py                          # default: no walls, difficulty 2
  python train_d3qn_per.py --wall_obstacles         # with walls
  python train_d3qn_per.py --resume weights_d3qn.pth
  python train_d3qn_per.py --episodes 5000 --wall_obstacles --render
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
ACTIONS = ("L45", "L22", "FW", "R22", "R45")
NUM_ACTIONS = len(ACTIONS)
OBS_DIM = 18
STACK_K = 4
INPUT_DIM = OBS_DIM * STACK_K  # 72

# Core DRL
GAMMA            = 0.99
LR               = 1e-4
BATCH_SIZE       = 256
REPLAY_CAPACITY  = 600_000
TRAIN_START      = 5_000
TRAIN_FREQ       = 8
TAU              = 0.005       # soft target update coefficient

# Exploration
EPS_START        = 1.0
EPS_END          = 0.10
EPS_DECAY_STEPS  = 8_000_000   # ~50% of total steps for 10000 episodes

# PER
PER_ALPHA        = 0.6         # prioritization exponent (0=uniform, 1=greedy)
PER_BETA_START   = 0.4         # IS correction exponent start
PER_BETA_END     = 1.0         # IS correction exponent end (anneal to 1.0)
PER_BETA_STEPS   = 8_000_000   # anneal beta over same horizon as epsilon
PER_EPS          = 1e-6        # small constant to avoid zero priority

# Reward normalization
REWARD_SCALE     = 50.0

# Training and Logging
SAVE_PATH        = "weights_d3qn.pth"
LOG_EVERY        = 25
DEFAULT_EPISODES = 3_000
MAX_STEPS_PER_EP = 1_000
PATIENCE = 10000   # episodes without improvement before stopping
MIN_EPISODES_BEFORE_STOP = 2000
EMA_ALPHA = 0.1   # smoothing factor

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
        self.tree = np.zeros(2 * capacity + 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write_ptr = 0
        self.n_entries = 0

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
        
        # Clamp to valid leaf range to guard against float rounding edge cases
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
        self.tree  = SumTree(capacity)
        self.alpha = alpha
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done) -> None:
        transition = (
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        # New transitions get max priority so they are sampled at least once
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
                lo    = segment * i
                hi    = segment * (i + 1)
                value = random.uniform(lo, hi)
                leaf_idx, priority, transition = self.tree.get(value)
                attempts += 1

            # Final fallback — scan for any non-None entry
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
            return None   # caller must handle

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

    Benefits for OBELIX:
      - Find phase: all actions equally bad → V(s) captures this, A stays near 0
      - Push phase: actions suddenly matter → A(s,a) captures directional bias
      - Faster convergence when most actions are equivalent (most of the episode)
    """

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Value stream V(s) → scalar
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Advantage stream A(s,a) → vector of size num_actions
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        V = self.value_stream(shared)                          # (B, 1)
        A = self.advantage_stream(shared)                      # (B, num_actions)
        Q = V + (A - A.mean(dim=1, keepdim=True))             # (B, num_actions)
        return Q


# ══════════════════════════════════════════════════════════════════════════════
# Frame Stack
# ══════════════════════════════════════════════════════════════════════════════
class FrameStack:
    def __init__(self, k: int, obs_dim: int):
        self.k = k
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
        arena_size=500,
        max_steps=MAX_STEPS_PER_EP,
        wall_obstacles=args.wall_obstacles,
        difficulty=3,
        box_speed=2,
        seed=0,
    )

    replay  = PrioritizedReplayBuffer(REPLAY_CAPACITY, PER_ALPHA)
    fstack  = FrameStack(STACK_K, OBS_DIM)

    # ── Tracking ──────────────────────────────────────────────────────────────
    total_steps      = 0
    episode_rewards  = []
    episode_successes = []
    losses            = []
    stuck_counts      = [] 
    best_mean        = -np.inf
    ema_reward   = None
    no_improve   = 0

    wall_str = "walls=ON" if args.wall_obstacles else "walls=OFF"
    print(f"\nTraining D3QN-PER | episodes={args.episodes} | {wall_str} | difficulty=3")
    print("=" * 70)

    rng_master = np.random.default_rng(42)   # reproducible master RNG
    
    for ep in range(1, args.episodes + 1):
        seed = int(rng_master.integers(10_000, 10_000_000))
        raw_obs = env.reset(seed=seed)
        state   = fstack.reset(raw_obs)

        ep_reward = 0.0
        done      = False

        # track stuck events per episode
        ep_stuck_count = 0

        while not done:
            # ε-greedy
            eps = get_epsilon(total_steps)
            if random.random() < eps:
                action_idx = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    action_idx = int(online_net(s_t).argmax(dim=1).item())

            action_str = ACTIONS[action_idx]
            raw_next, reward, done = env.step(action_str, render=args.render)
            next_state = fstack.step(raw_next)

            # Count stuck events 
            if raw_next[17] == 1:
                ep_stuck_count += 1

            # Store normalized reward, log raw
            
            wall_hit = bool(raw_next[17] == 1)
            if wall_hit and not env.enable_push:
                extra_penalty = -50 if args.wall_obstacles else -300   # extra -300 internally
                shaped = reward + extra_penalty
            else:
                shaped = reward
                
            stored_reward = shaped / REWARD_SCALE
            
            last_data_idx = replay.tree.write_ptr
            replay.push(state, action_idx, stored_reward, next_state, done)

            state      = next_state
            ep_reward += reward          # raw reward for honest logging
            total_steps += 1

            # ── Gradient update ───────────────────────────────────────────────
            if (
                len(replay) >= TRAIN_START
                and total_steps % TRAIN_FREQ == 0
            ):
                beta = get_beta(total_steps)
                result = replay.sample(BATCH_SIZE, beta)
                if result is None:
                    continue   # skip this update if sample failed
                (
                    s, a, r, s2, d,
                    leaf_idxs, is_weights
                ) = replay.sample(BATCH_SIZE, beta)

                s, a, r, s2, d = (
                    s.to(device), a.to(device), r.to(device),
                    s2.to(device), d.to(device),
                )
                is_weights = is_weights.to(device)

                # DDQN target: online selects action, target evaluates it
                with torch.no_grad():
                    best_actions = online_net(s2).argmax(dim=1, keepdim=True)
                    q_next       = target_net(s2).gather(1, best_actions).squeeze(1)
                    q_target     = r + GAMMA * q_next * (1.0 - d)

                q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                # TD errors for priority update
                td_errors = (q_pred - q_target).detach().cpu().numpy()
                replay.update_priorities(leaf_idxs, td_errors)

                # IS-weighted Huber loss
                element_loss = nn.functional.smooth_l1_loss(
                    q_pred, q_target, reduction="none"
                )
                loss = (is_weights * element_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                # Soft target update every gradient step
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

            # Retrieve and update the last transition's stored data
            data_idx = last_data_idx
            data_idx = int(np.clip(data_idx, 0, replay.tree.capacity - 1))

            if replay.tree.data[data_idx] is not None:
                s, a, r, s2, d = replay.tree.data[data_idx]
                replay.tree.data[data_idx] = (s, a, r + efficiency_bonus, s2, d)
        
        episode_rewards.append(ep_reward)
        episode_successes.append(success)
        stuck_counts.append(ep_stuck_count)

        # ── Periodic priority refresh ──────────────────────────────────────────
        if ep % 100 == 0:
            n             = replay.tree.n_entries
            refresh_count = n // 50                    # 2% of buffer
            oldest_start  = replay.tree.write_ptr % replay.tree.capacity
            for i in range(refresh_count):
                idx      = (oldest_start + i) % replay.tree.capacity
                
                # Only refresh slots that actually have data
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
            window  = min(LOG_EVERY, len(episode_rewards))
            mean_r  = np.mean(episode_rewards[-window:])
            mean_s  = np.mean(episode_successes[-window:]) * 100
            mean_l  = np.mean(losses[-200:]) if losses else 0.0
            eps_now = get_epsilon(total_steps)
            beta_now = get_beta(total_steps)
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

            
            
            # if mean_r > best_mean:
            #     best_mean = mean_r
            if ema_reward > best_mean:
                best_mean  = ema_reward
                no_improve = 0
                torch.save(online_net.state_dict(), SAVE_PATH)
                print(f"  ✓ New best (ema_r={best_mean:.1f}) → {SAVE_PATH}")
            
            # Stop early if mean reward is not improving
            else:
                no_improve += LOG_EVERY
                if no_improve >= PATIENCE and ep >= MIN_EPISODES_BEFORE_STOP:
                    print(f"Early stopping at episode {ep} — no improvement in {PATIENCE} episodes")
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
    parser.add_argument("--episodes",      type=int,  default=DEFAULT_EPISODES)
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--wall_obstacles",action="store_true")
    parser.add_argument("--resume",        type=str,  default=None)
    args = parser.parse_args()
    train(args)
