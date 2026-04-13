"""
DQN Training Script for OBELIX Environment
==========================================
Difficulty: 0 (static box), wall_obstacles=True

Architecture:
  - MLP: 18 → 256 → 256 → 5
  - Replay buffer (uniform)
  - Target network (hard update every C steps)
  - Epsilon-greedy exploration with linear decay
  - Frame stacking (k=4) to handle partial observability

Usage:
  python train_dqn.py
  python train_dqn.py --episodes 3000 --render
  python train_dqn.py --resume weights_dqn.pth
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

# ── Make sure obelix.py is importable ─────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX

# ══════════════════════════════════════════════════════════════════════════════
# Hyper-parameters
# ══════════════════════════════════════════════════════════════════════════════
ACTIONS = ("L45", "L22", "FW", "R22", "R45")
NUM_ACTIONS = len(ACTIONS)

OBS_DIM = 18
STACK_K = 4                    # number of frames to stack
INPUT_DIM = OBS_DIM * STACK_K  # 72

GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
TARGET_UPDATE_FREQ = 500       # steps between hard target-net syncs
TRAIN_START = 2_000            # steps before first gradient update
TRAIN_FREQ = 4                 # gradient update every N env steps

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 2000_000      # linear decay over this many env steps

MAX_STEPS_PER_EP = 1_000
DEFAULT_EPISODES = 2_000

SAVE_PATH = "weights_dqn.pth"
LOG_EVERY = 50                 # print stats every N episodes


# ══════════════════════════════════════════════════════════════════════════════
# Neural Network
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
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,                dtype=torch.long),
            torch.tensor(rewards,                dtype=torch.float32),
            torch.tensor(np.array(next_states),  dtype=torch.float32),
            torch.tensor(dones,                  dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════
# Frame Stack
# ══════════════════════════════════════════════════════════════════════════════
class FrameStack:
    """Maintains a rolling window of k observations and returns their concat."""
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
# Epsilon schedule
# ══════════════════════════════════════════════════════════════════════════════
def get_epsilon(step: int) -> float:
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Networks ──────────────────────────────────────────────────────────────
    online_net = DQN(INPUT_DIM, NUM_ACTIONS).to(device)
    target_net = DQN(INPUT_DIM, NUM_ACTIONS).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    if args.resume and os.path.exists(args.resume):
        online_net.load_state_dict(
            torch.load(args.resume, map_location=device)
        )
        target_net.load_state_dict(online_net.state_dict())
        print(f"Resumed from {args.resume}")

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()  # Huber loss

    # ── Environment ───────────────────────────────────────────────────────────
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=MAX_STEPS_PER_EP,
        wall_obstacles=True,
        difficulty=0,
        box_speed=2,
        seed=0,
    )

    replay = ReplayBuffer(REPLAY_CAPACITY)
    frame_stack = FrameStack(STACK_K, OBS_DIM)

    # ── Tracking ──────────────────────────────────────────────────────────────
    total_steps = 0
    episode_rewards = []
    episode_successes = []
    losses = []
    best_mean = -np.inf

    print(f"\nTraining DQN | episodes={args.episodes} | wall_obstacles=True | difficulty=0")
    print("=" * 70)

    for ep in range(1, args.episodes + 1):
        seed = ep  # different seed each episode for diversity
        raw_obs = env.reset(seed=seed)
        state = frame_stack.reset(raw_obs)

        ep_reward = 0.0
        success = False
        done = False

        while not done:
            # ε-greedy action selection
            eps = get_epsilon(total_steps)
            if random.random() < eps:
                action_idx = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    action_idx = int(online_net(s_t).argmax(dim=1).item())

            action_str = ACTIONS[action_idx]
            raw_next, reward, done = env.step(action_str, render=args.render)
            next_state = frame_stack.step(raw_next)

            # Track success (large positive reward spike = success bonus)
            if reward >= 2000:
                success = True

            # Clip reward for stability (keep the big bonuses but clip penalties)
            clipped_reward = np.clip(reward, -200, 2000)

            replay.push(state, action_idx, clipped_reward, next_state, done)

            state = next_state
            ep_reward += reward
            total_steps += 1

            # ── Gradient update ───────────────────────────────────────────────
            if (
                len(replay) >= TRAIN_START
                and total_steps % TRAIN_FREQ == 0
            ):
                s, a, r, s2, d = replay.sample(BATCH_SIZE)
                s, a, r, s2, d = (
                    s.to(device), a.to(device), r.to(device),
                    s2.to(device), d.to(device),
                )

                with torch.no_grad():
                    q_next = target_net(s2).max(dim=1)[0]
                    q_target = r + GAMMA * q_next * (1.0 - d)

                q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = loss_fn(q_pred, q_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()
                losses.append(loss.item())

            # ── Target net sync ───────────────────────────────────────────────
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())

        episode_rewards.append(ep_reward)
        episode_successes.append(success)

        # ── Logging ───────────────────────────────────────────────────────────
        if ep % LOG_EVERY == 0:
            window = min(LOG_EVERY, len(episode_rewards))
            mean_r = np.mean(episode_rewards[-window:])
            mean_s = np.mean(episode_successes[-window:]) * 100
            mean_l = np.mean(losses[-500:]) if losses else 0.0
            eps_now = get_epsilon(total_steps)

            print(
                f"Ep {ep:5d}/{args.episodes} | "
                f"steps={total_steps:7d} | "
                f"mean_r={mean_r:8.1f} | "
                f"success={mean_s:5.1f}% | "
                f"loss={mean_l:.4f} | "
                f"eps={eps_now:.3f}"
            )

            # Save best model
            if mean_r > best_mean:
                best_mean = mean_r
                torch.save(online_net.state_dict(), SAVE_PATH)
                print(f"  ✓ New best model saved → {SAVE_PATH}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = "weights_dqn_final.pth"
    torch.save(online_net.state_dict(), final_path)
    print(f"\nTraining complete. Final model → {final_path}")
    print(f"Best model (mean_r={best_mean:.1f}) → {SAVE_PATH}")

    return online_net


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)
