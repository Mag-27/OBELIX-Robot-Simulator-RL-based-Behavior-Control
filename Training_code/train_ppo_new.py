"""
PPO trainer for OBELIX
Updates:
  • Curricula: Alternates 10 episodes (Walls ON) and 10 episodes (Walls OFF).
  • Rendering: Runs 1 rendered evaluation episode every 10 training episodes.
  • LSTM + Heuristic Modes + Reward Shaping integration.
"""

from __future__ import annotations
import argparse
import random
import time
import os
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
ACTIONS    = ["L45", "L22", "FW", "R22", "R45"]
FW_IDX     = 2 
ROT_IDX    = [0, 1, 3, 4]
N_OBS_RAW  = 18
N_OBS      = 22
N_ACTIONS  = 5
N_MODES    = 4

MODE_SEARCH, MODE_ALIGN, MODE_PUSH, MODE_UNSTUCK = 0, 1, 2, 3
MAX_POLICY_GRAD, MAX_VALUE_GRAD = 0.5, 0.5
SONAR_POINT_ANGLES = np.deg2rad([-90, -90, 0, 0, 0, 0, 90, 90])

# ─────────────────────────────────────────────────────────────────────────────
#  Utility & Observation Processing
# ─────────────────────────────────────────────────────────────────────────────
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_mode(obs: np.ndarray) -> int:
    if obs[17]: return MODE_UNSTUCK
    if obs[16]: return MODE_PUSH
    if np.any(obs[0:16]): return MODE_ALIGN
    return MODE_SEARCH

def prep_obs(raw_obs: np.ndarray) -> np.ndarray:
    mode = get_mode(raw_obs)
    oh = np.zeros(N_MODES, dtype=np.float32)
    oh[mode] = 1.0
    aug = np.concatenate([raw_obs, oh])
    return (aug - aug.mean()) / (aug.std() + 1e-8)

def linear_decay(initial: float, final: float, current: int, total: int) -> float:
    t = min(current / max(total - 1, 1), 1.0)
    return initial * (1.0 - t) + final * t

# ─────────────────────────────────────────────────────────────────────────────
#  Reward Shaper
# ─────────────────────────────────────────────────────────────────────────────
class RewardShaper:
    def __init__(self):
        self.reset()
    def reset(self):
        self._had_contact = False
        self._prev_ir = 0.0
        self._consec_stuck = 0
    def shape(self, raw_next, action, env_r):
        ir, stuck = float(raw_next[16]), float(raw_next[17])
        r = 0.0
        if stuck:
            self._consec_stuck += 1
            r -= 0.2 + (0.05 * self._consec_stuck)
        else:
            self._consec_stuck = 0
        if ir > 0.5 and not self._had_contact:
            r += 0.5
            self._had_contact = True
        return r

# ─────────────────────────────────────────────────────────────────────────────
#  Networks
# ─────────────────────────────────────────────────────────────────────────────
_MODE_LOGIT_BIAS = torch.tensor([
    [0.1, 0.1, -0.1, 0.1, 0.1], [0.2, 0.2, -0.2, 0.2, 0.2],
    [-0.1, -0.1, 0.4, -0.1, -0.1], [0.3, 0.3, -0.4, 0.3, 0.3]
], dtype=torch.float32)

class PolicyNetwork(nn.Module):
    def __init__(self, stateDim=N_OBS, n_actions=N_ACTIONS, hDims=(128,)):
        super().__init__()
        self.inputLayer = nn.Linear(stateDim, hDims[0])
        self.lstm = nn.LSTM(hDims[0], hDims[0], batch_first=True)
        self.out = nn.Linear(hDims[0], n_actions)
        self.register_buffer("mode_logit_bias", _MODE_LOGIT_BIAS.clone())

    def forward(self, states, actions=None, hidden=None):
        x = states if isinstance(states, torch.Tensor) else torch.tensor(states, dtype=torch.float32)
        if x.dim() == 2: x = x.unsqueeze(1)
        feat = torch.relu(self.inputLayer(x))
        feat, hidden = self.lstm(feat, hidden)
        logits = self.out(feat)
        # Apply mode bias
        mode_oh = x[..., N_OBS_RAW:]
        bias = torch.einsum("...m,mn->...n", mode_oh, self.mode_logit_bias)
        probs = torch.softmax(logits + bias, dim=-1)
        dist = Categorical(probs=probs)
        if actions is None: a = dist.sample()
        else: a = actions
        return a, dist.log_prob(a), dist.entropy(), probs.argmax(-1), hidden, probs

class ValueNetwork(nn.Module):
    def __init__(self, stateDim=N_OBS, hDims=(128,)):
        super().__init__()
        self.inputLayer = nn.Linear(stateDim, hDims[0])
        self.lstm = nn.LSTM(hDims[0], hDims[0], batch_first=True)
        self.out = nn.Linear(hDims[0], 1)
    def forward(self, states, hidden=None):
        x = states if isinstance(states, torch.Tensor) else torch.tensor(states, dtype=torch.float32)
        if x.dim() == 2: x = x.unsqueeze(1)
        feat, hidden = self.lstm(torch.relu(self.inputLayer(x)), hidden)
        return self.out(feat).squeeze(-1), hidden

# ─────────────────────────────────────────────────────────────────────────────
#  Buffer
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeBuffer:
    def __init__(self, gamma, lam, stateDim, numWorkers, maxEpisodes, maxSteps):
        self.gamma, self.lam = gamma, lam
        self.stateDim, self.numWorkers = stateDim, numWorkers
        self.maxEpisodes, self.maxSteps = maxEpisodes, maxSteps
        self.reset()

    def reset(self):
        self.bufferStates = np.zeros((self.maxEpisodes, self.maxSteps, self.stateDim), dtype=np.float32)
        self.bufferActions = np.zeros((self.maxEpisodes, self.maxSteps), dtype=np.int64)
        self.bufferReturns = np.zeros((self.maxEpisodes, self.maxSteps), dtype=np.float32)
        self.bufferGAEs    = np.zeros((self.maxEpisodes, self.maxSteps), dtype=np.float32)
        self.bufferLogps   = np.zeros((self.maxEpisodes, self.maxSteps), dtype=np.float32)
        self.episodeRewards = np.zeros(self.maxEpisodes)
        self.episodeSteps   = np.zeros(self.maxEpisodes, dtype=np.int32)
        self.curr_ep_ids    = list(range(self.numWorkers))

    def fill(self, envs, pNet, vNet, args, total_eps_global, rng):
        pNet.eval(); vNet.eval()
        w_steps = np.zeros(self.numWorkers, dtype=np.int32)
        w_rewards = np.zeros((self.numWorkers, self.maxSteps))
        h_p = [None]*self.numWorkers
        h_v = [None]*self.numWorkers
        shapers = [RewardShaper() for _ in range(self.numWorkers)]
        
        raw_obs = [np.array(e.reset(), dtype=np.float32) for e in envs]
        obs = np.stack([prep_obs(r) for r in raw_obs])
        
        filled_count = self.numWorkers
        while filled_count < self.maxEpisodes:
            with torch.no_grad():
                tensor_obs = torch.from_numpy(obs).float().unsqueeze(1)
                a_net, lp_net, _, _, next_hp, _ = pNet(tensor_obs, hidden=None) # Simplification: no step-LSTM during rollout for speed
                v_net, _ = vNet(tensor_obs)

            for i in range(self.numWorkers):
                ep_idx = self.curr_ep_ids[i]
                t = w_steps[i]
                
                # Exploration
                eps = linear_decay(args.eps_start, args.eps_end, total_eps_global, args.total_episodes)
                act = int(a_net[i].item()) if rng.rand() > eps else rng.randint(N_ACTIONS)
                
                next_raw, r_env, done = envs[i].step(ACTIONS[act])
                r_shaped = r_env + shapers[i].shape(next_raw, act, r_env)
                
                self.bufferStates[ep_idx, t] = obs[i]
                self.bufferActions[ep_idx, t] = act
                self.bufferLogps[ep_idx, t] = lp_net[i].item()
                w_rewards[i, t] = r_shaped
                
                obs[i] = prep_obs(next_raw)
                w_steps[i] += 1
                
                if done or w_steps[i] >= self.maxSteps:
                    # GAE/Return calculation (simplified)
                    self.episodeSteps[ep_idx] = w_steps[i]
                    self.episodeRewards[ep_idx] = w_rewards[i, :w_steps[i]].sum()
                    
                    # Compute Returns
                    ret = 0
                    for rev_t in reversed(range(w_steps[i])):
                        ret = w_rewards[i, rev_t] + self.gamma * ret
                        self.bufferReturns[ep_idx, rev_t] = ret
                    
                    # Swap worker to new episode
                    if filled_count < self.maxEpisodes:
                        self.curr_ep_ids[i] = filled_count
                        filled_count += 1
                        w_steps[i] = 0
                        w_rewards[i, :] = 0
                        shapers[i].reset()
                        raw_obs[i] = np.array(envs[i].reset(), dtype=np.float32)
                        obs[i] = prep_obs(raw_obs[i])
                    else:
                        break

    def get_data(self):
        valid = self.episodeSteps > 0
        return (self.bufferStates[valid], self.bufferActions[valid], 
                self.bufferReturns[valid], self.bufferLogps[valid], self.episodeSteps[valid])

# ─────────────────────────────────────────────────────────────────────────────
#  PPO Engine
# ─────────────────────────────────────────────────────────────────────────────
class PPO:
    def __init__(self, OBELIX, args):
        self.args = args
        self.OBELIX = OBELIX
        self.rng = np.random.RandomState(args.seed)
        set_global_seeds(args.seed)
        
        self.pNet = PolicyNetwork()
        self.vNet = ValueNetwork()
        self.pOpt = optim.Adam(self.pNet.parameters(), lr=args.lr_policy)
        self.vOpt = optim.Adam(self.vNet.parameters(), lr=args.lr_value)
        
        self.buffer = EpisodeBuffer(args.gamma, args.lam, N_OBS, args.num_workers, args.episodes_per_fill, args.max_steps)
        self.total_eps = 0

    def make_envs(self, walls):
        return [self.OBELIX(wall_obstacles=walls, seed=self.args.seed+i, max_steps=self.args.max_steps) for i in range(self.args.num_workers)]

    def train(self):
        use_walls = False
        envs = self.make_envs(use_walls)
        
        while self.total_eps < self.args.total_episodes:
            # 1. Logic: Toggle walls every 10 episodes
            new_wall_state = (self.total_eps // 10) % 2 == 1
            if new_wall_state != use_walls:
                use_walls = new_wall_state
                for e in envs: e.close()
                envs = self.make_envs(use_walls)
                print(f"\n[Curriculum] Switch: Walls {'ON' if use_walls else 'OFF'}")

            # 2. Render Check: Every 10 training episodes
            if self.total_eps % 10 == 0 and self.total_eps > 0:
                print(f"--- Rendering Eval Episode at {self.total_eps} ---")
                self.evaluate(render=True)

            # 3. Fill & Update
            self.buffer.fill(envs, self.pNet, self.vNet, self.args, self.total_eps, self.rng)
            self.update_networks()
            
            avg_r = np.mean(self.buffer.episodeRewards[self.buffer.episodeSteps > 0])
            self.total_eps += self.args.episodes_per_fill
            print(f"Step {self.total_eps} | Avg Reward: {avg_r:.2f}")
            self.buffer.reset()

    def update_networks(self):
        states, actions, returns, old_logps, steps = self.buffer.get_data()
        self.pNet.train(); self.vNet.train()
        
        for _ in range(self.args.policy_epochs):
            # Flatten for PPO update
            s_t = torch.tensor(states).float()
            a_t = torch.tensor(actions).long()
            r_t = torch.tensor(returns).float()
            lp_t = torch.tensor(old_logps).float()
            
            _, new_lp, entropy, _, _, _ = self.pNet(s_t, a_t)
            v_pred, _ = self.vNet(s_t)
            
            adv = r_t - v_pred.detach()
            ratio = torch.exp(new_lp - lp_t)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.args.policy_clip, 1+self.args.policy_clip) * adv
            
            p_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            v_loss = F.mse_loss(v_pred, r_t)
            
            self.pOpt.zero_grad(); p_loss.backward(); self.pOpt.step()
            self.vOpt.zero_grad(); v_loss.backward(); self.vOpt.step()

    def evaluate(self, render=False):
        env = self.OBELIX(wall_obstacles=True, seed=999, max_steps=self.args.max_steps)
        raw = env.reset()
        total_r = 0
        self.pNet.eval()
        for _ in range(self.args.max_steps):
            obs = prep_obs(np.array(raw, dtype=np.float32))
            with torch.no_grad():
                _, _, _, greedy, _, _ = self.pNet(torch.from_numpy(obs).float().view(1,1,-1))
            raw, r, done = env.step(ACTIONS[greedy.item()], render=render)
            total_r += r
            if done: break
        env.close()
        print(f"Eval Reward: {total_r}")

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--episodes_per_fill", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr_policy", type=float, default=3e-4)
    parser.add_argument("--lr_value", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--policy_epochs", type=int, default=5)
    parser.add_argument("--policy_clip", type=float, default=0.2)
    parser.add_argument("--eps_start", type=float, default=0.2)
    parser.add_argument("--eps_end", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Dynamic Import
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", args.obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    trainer = PPO(mod.OBELIX, args)
    trainer.train()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()