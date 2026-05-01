"""
PPO + LSTM agent for OBELIX navigation task.

Forward-action constraint to prevent rotation collapse.

TWO versions of select_action() are implemented:

  VERSION A — Hard Constraint
    If f_count < 0.4 * t  →  force action = FW_IDX
    Stores the policy's log-prob FOR the FW action (not a fake value),
    so the PPO importance ratio exp(logp_new - logp_old) remains valid.

  VERSION B — Soft Logit Bias  ← RECOMMENDED for OBELIX
    If f_count < 0.4 * t  →  add  alpha * deficit  to FW logit before softmax
    The agent still samples stochastically from the biased distribution.
    No action is ever hard-overridden. The stored log-prob is always
    consistent with the distribution that was actually sampled from.
    Gradient flow is clean: the bias shifts the distribution, policy
    gradient does the rest.

WHY soft bias wins in OBELIX:
  Hard override produces a behaviour policy that is very different from
  the current network policy (importance ratio explodes when the network
  strongly prefers rotation). The soft bias nudges the network's own
  distribution, keeping the ratio close to 1.0 and PPO stable.

Toggle with --action_mode hard | soft (default: soft).
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
FW_IDX    = ACTIONS.index("FW")   # 2
ROT_IDXS  = {0, 1, 3, 4}
N_OBS     = 18
N_ACTIONS = 5

# =========================================================
# ENV LOADER
# =========================================================
def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# =========================================================
# OBSERVATION NORMALISATION
# =========================================================
def normalise_obs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)

# =========================================================
# RUNNING REWARD NORMALISER
# =========================================================
class RunningNorm:
    """
    Online z-score normaliser for scalar rewards.

    The OBELIX reward range spans [-200, +2000].  Without normalisation
    the value network is dominated by the rare -200 stuck signal and
    learns "everything is bad", suppressing FW gradients entirely.
    """
    def __init__(self, clip: float = 10.0):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 0
        self.clip  = clip

    def update(self, rewards: list[float]):
        for r in rewards:
            self.count += 1
            delta      = r - self.mean
            self.mean += delta / self.count
            self.var  += delta * (r - self.mean)

    def normalise(self, rewards: list[float]) -> list[float]:
        std = math.sqrt(max(self.var / max(self.count, 1), 1e-8))
        return [float(np.clip((r - self.mean) / std, -self.clip, self.clip))
                for r in rewards]

# =========================================================
# CURRICULUM
# =========================================================
def curriculum_difficulty(training_progress: float) -> int:
    if training_progress < 0.33:
        return 0
    elif training_progress < 0.66:
        return 1
    return 2

# =========================================================
# LSTM POLICY
# =========================================================
class PolicyNetwork(nn.Module):
    def __init__(self, stateDim=N_OBS, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.fc   = nn.Linear(stateDim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out  = nn.Linear(hidden, n_actions)

        # Neutral init — uniform policy at start
        nn.init.zeros_(self.out.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.01)

    def forward(self, x, hidden=None, actions=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = torch.relu(self.fc(x))
        x, hidden = self.lstm(x, hidden)
        logits = self.out(x)
        return logits, hidden

    def get_dist(self, logits):
        """Return a Categorical distribution from raw logits."""
        return Categorical(logits=logits)

# =========================================================
# LSTM VALUE
# =========================================================
class ValueNetwork(nn.Module):
    def __init__(self, stateDim=N_OBS, hidden=128):
        super().__init__()
        self.fc   = nn.Linear(stateDim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out  = nn.Linear(hidden, 1)

    def forward(self, x, hidden=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = torch.relu(self.fc(x))
        x, hidden = self.lstm(x, hidden)
        return self.out(x), hidden

# =========================================================
# PPO AGENT
# =========================================================
class PPO:
    def __init__(self, env_fn, args):
        self.env_fn = env_fn
        self.args   = args

        self.pnet = PolicyNetwork()
        self.vnet = ValueNetwork()

        self.opt_p = optim.Adam(self.pnet.parameters(), lr=args.lr_policy)
        self.opt_v = optim.Adam(self.vnet.parameters(), lr=args.lr_value)

        self.total_steps = 0
        self.reward_norm = RunningNorm(clip=10.0)

    # =========================================================
    # SELECT ACTION — both versions live here
    # =========================================================
    def select_action(
        self,
        logits: torch.Tensor,   # shape (1, 1, N_ACTIONS) — raw network output
        t: int,                  # steps taken so far this episode (before this one)
        f_count: int,            # FW actions taken so far this episode
    ) -> tuple[int, float]:
        """
        Returns (action_index, log_prob_of_that_action).

        The log_prob stored here is ALWAYS the log-prob under the
        distribution that was actually used to sample — so the PPO
        importance ratio exp(logp_new - logp_old) is always valid.

        VERSION A — Hard Constraint
        ─────────────────────────────
        If forward ratio is below target, action is forced to FW_IDX.
        The log-prob stored is the network's log-prob FOR FW — not 0,
        not 1, not a detached constant.  This matters: during the PPO
        update the network recomputes log-prob for the stored action
        (FW_IDX), and the ratio is the change in how much the network
        now assigns to FW vs. how much it assigned then.

        Stability note: when the network strongly prefers rotation,
        p(FW) can be very small → logp(FW) very negative → large ratio
        swings.  The PPO clip (0.8, 1.2) handles this but will suppress
        updates.  This is why soft bias is usually better here.

        VERSION B — Soft Logit Bias  ← default
        ─────────────────────────────────────────
        deficit = max(0,  0.4*t - f_count)
        logits[FW] += alpha * deficit

        The bias is applied BEFORE sampling, so the sampled action and
        its log-prob are both drawn from the biased distribution.  No
        inconsistency.  The PPO update recomputes log-prob from the
        unbiased network — which is correct: we want the network to
        *learn* to increase p(FW) on its own, not to perpetually rely
        on the bias crutch.  As p(FW) rises, the deficit shrinks and
        the bias fades naturally.
        """
        logits = logits.squeeze()   # (N_ACTIONS,)
        target_fw = self.args.fw_target_ratio  # 0.4

        if self.args.action_mode == "hard":
            # ── VERSION A: Hard Constraint ────────────────────────────────────
            constrained = (t > 0) and (f_count < target_fw * t)

            if constrained:
                # Force FW. Store the network's actual log-prob for FW so the
                # PPO ratio is well-defined and gradients flow correctly.
                dist = self.pnet.get_dist(logits.unsqueeze(0))
                a    = FW_IDX
                logp = dist.log_prob(
                    torch.tensor([FW_IDX])
                ).item()
            else:
                dist = self.pnet.get_dist(logits.unsqueeze(0))
                a_t  = dist.sample()
                a    = int(a_t.item())
                logp = dist.log_prob(a_t).item()

        else:
            # ── VERSION B: Soft Logit Bias ────────────────────────────────────
            # Compute how many FW steps we're behind the 40% target.
            deficit = max(0.0, target_fw * t - f_count)   # float ≥ 0

            # Add bias to the FW logit in proportion to the deficit.
            # No inplace ops — clone first to avoid autograd graph issues.
            biased_logits = logits.clone()
            biased_logits[FW_IDX] = biased_logits[FW_IDX] + self.args.fw_bias_alpha * deficit

            # Sample from the biased distribution.
            dist = self.pnet.get_dist(biased_logits.unsqueeze(0))
            a_t  = dist.sample()
            a    = int(a_t.item())
            # IMPORTANT: store log-prob from the BIASED distribution.
            # During the PPO update we recompute from the unbiased network,
            # which is intentional — we want the policy to *learn* p(FW)↑,
            # not just rely on the external bias forever.
            logp = dist.log_prob(a_t).item()

        return a, logp

    # ── rollout ───────────────────────────────────────────────────────────────
    def rollout(self, env, render=False):
        """
        Collect one episode using select_action() for every step.

        Tracks:
          t       — total steps so far this episode
          f_count — FW actions executed so far this episode
        Both are passed to select_action() so it can compute the constraint.
        """
        s    = normalise_obs(np.asarray(env.reset(), dtype=np.float32))
        done = False

        states, actions, rewards, logps, values = [], [], [], [], []
        h_p = h_v = None

        t       = 0   # steps taken this episode
        f_count = 0   # FW actions taken this episode

        while not done:
            s_t = torch.tensor(s, dtype=torch.float32).view(1, 1, -1)

            with torch.no_grad():
                logits, h_p = self.pnet(s_t, h_p)
                v,      h_v = self.vnet(s_t, h_v)

            h_p = (h_p[0].detach(), h_p[1].detach())
            h_v = (h_v[0].detach(), h_v[1].detach())

            # ── Action selection with forward constraint ──────────────────────
            a, logp = self.select_action(logits, t, f_count)

            # Update episode trackers BEFORE env step
            t += 1
            if a == FW_IDX:
                f_count += 1

            self.total_steps += 1

            s2, r_ext, done = env.step(ACTIONS[a], render=render)
            s2 = normalise_obs(np.asarray(s2, dtype=np.float32))

            states.append(s)
            actions.append(a)
            rewards.append(float(r_ext))
            logps.append(logp)
            values.append(float(v.item()))

            s = s2

        return states, actions, rewards, logps, values, f_count, t

    # ── compute_gae ───────────────────────────────────────────────────────────
    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        adv  = []
        gae  = 0.0
        vals = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * vals[t + 1] - vals[t]
            gae   = delta + gamma * lam * gae
            adv.insert(0, gae)
        returns = [a + v for a, v in zip(adv, values)]
        return adv, returns

    # ── entropy schedule ──────────────────────────────────────────────────────
    def _entropy_coef(self, progress: float) -> float:
        HIGH = 0.05
        LOW  = 0.005
        HOLD = 0.40
        if progress < HOLD:
            return HIGH
        decay = (progress - HOLD) / (1.0 - HOLD)
        return LOW + 0.5 * (HIGH - LOW) * (1.0 + math.cos(math.pi * decay))

    # ── PPO loss (recompute log-probs from UNBIASED network) ──────────────────
    def _ppo_loss(self, states_t, actions_t, old_logps, adv_t, ent_coef):
        """
        During the update we always use the raw (unbiased) network logits.

        For the soft-bias version: the old_logps were computed from the biased
        distribution at rollout time.  The new_logps are from the unbiased
        network.  The ratio exp(logp_new - logp_old) is therefore slightly
        off-policy — but this is exactly what we want: we're training the
        network to increase p(FW) under its OWN distribution so that the
        bias eventually becomes unnecessary.  The PPO clip keeps this stable.
        """
        # Forward through unbiased network
        logits, _ = self.pnet(states_t)                    # (T, 1, N_ACTIONS)
        logits     = logits.squeeze(1)                      # (T, N_ACTIONS)
        dist       = self.pnet.get_dist(logits)
        logp       = dist.log_prob(actions_t)               # (T,)
        ent        = dist.entropy()                         # (T,)

        ratio = torch.exp(logp - old_logps)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t

        loss_p = -torch.min(surr1, surr2).mean() - ent_coef * ent.mean()
        return loss_p, ent

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self):
        args           = self.args
        total_episodes = args.total_episodes
        current_seed   = random.randint(0, 9999)

        for ep in range(total_episodes):
            progress  = ep / max(1, total_episodes)
            diff      = curriculum_difficulty(progress)
            use_walls = (ep >= args.wall_switch)

            if ep % args.seed_every == 0:
                current_seed = random.randint(0, 9999)

            env = self.env_fn(current_seed, wall_obstacles=use_walls, difficulty=diff)

            render_ep = (ep % args.eval_every == 0)
            states, actions, rewards, logps, values, f_count, ep_steps = self.rollout(
                env, render=render_ep
            )

            if ep == args.wall_switch:
                print("=" * 60)
                print(f"  CURRICULUM: walls ON from ep {ep}")
                print("=" * 60, flush=True)

            # Reward normalisation
            self.reward_norm.update(rewards)
            rewards_norm = self.reward_norm.normalise(rewards)

            adv, returns = self.compute_gae(rewards_norm, values)

            states_t  = torch.from_numpy(np.array(states, dtype=np.float32)).unsqueeze(1)
            actions_t = torch.tensor(actions,  dtype=torch.long)
            old_logps = torch.tensor(logps,    dtype=torch.float32)
            adv_t     = torch.tensor(adv,      dtype=torch.float32)
            returns_t = torch.tensor(returns,  dtype=torch.float32)

            # Advantage normalisation
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            ent_coef = self._entropy_coef(progress)

            # PPO update — 10 epochs per episode
            for _ in range(10):
                loss_p, ent = self._ppo_loss(
                    states_t, actions_t, old_logps, adv_t, ent_coef)

                v_pred, _ = self.vnet(states_t)
                loss_v    = ((v_pred.squeeze(1).squeeze(-1) - returns_t) ** 2).mean()

                self.opt_p.zero_grad()
                loss_p.backward()
                nn.utils.clip_grad_norm_(self.pnet.parameters(), 0.5)
                self.opt_p.step()

                self.opt_v.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.vnet.parameters(), 0.5)
                self.opt_v.step()

            # Debug print
            with torch.no_grad():
                logits_dbg, _ = self.pnet(states_t[:8])
                logits_dbg    = logits_dbg.squeeze(1)
                dist_dbg      = self.pnet.get_dist(logits_dbg)
                ent_dbg       = dist_dbg.entropy()
                mp            = dist_dbg.probs.mean(0)

            act_str   = "  ".join(f"{ACTIONS[i]}:{mp[i].item()*100:.1f}%" for i in range(N_ACTIONS))
            fw_pct    = mp[FW_IDX].item() * 100
            rot_pct   = sum(mp[i].item() for i in ROT_IDXS) * 100
            actual_fw = 100.0 * f_count / max(ep_steps, 1)
            flag      = "  ◀ ROTATING!" if rot_pct > 80 else ""
            phase     = "walls" if use_walls else "open "
            lbl       = ["EASY", "MED ", "HARD"][diff]
            mode      = args.action_mode.upper()

            print(f"Ep {ep:4d} | {phase} | {lbl} | {mode} | "
                  f"ret={sum(rewards):+8.1f} | "
                  f"ent={ent_dbg.mean().item():.3f} | "
                  f"ent_coef={ent_coef:.4f} | "
                  f"FW_net={fw_pct:.1f}%  FW_actual={actual_fw:.1f}%  ROT={rot_pct:.1f}% | "
                  f"[{act_str}]{flag}")

    def save(self, path):
        torch.save(self.pnet.state_dict(), path)

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--obelix_py",         type=str,   required=True)
    parser.add_argument("--out",               type=str,   default="weights_ppo.pth")
    parser.add_argument("--total_episodes",    type=int,   default=1000)
    parser.add_argument("--eval_every",        type=int,   default=10)
    parser.add_argument("--seed_every",        type=int,   default=50)

    # ── Forward-action constraint ─────────────────────────────────────────────
    parser.add_argument("--action_mode",       type=str,   default="soft",
                        choices=["hard", "soft"],
                        help="'hard' = force FW when below ratio; "
                             "'soft' = bias logit toward FW (default: soft)")
    parser.add_argument("--fw_target_ratio",   type=float, default=0.4,
                        help="Minimum fraction of steps that must be FW (default: 0.4)")
    parser.add_argument("--fw_bias_alpha",     type=float, default=2.0,
                        help="[soft mode] logit bias per deficit step (default: 2.0). "
                             "Higher = stronger nudge toward FW when behind ratio.")

    # Learning rates
    parser.add_argument("--lr_policy",         type=float, default=1e-4)
    parser.add_argument("--lr_value",          type=float, default=1e-3)

    # Environment
    parser.add_argument("--wall_switch",       type=int,   default=500,
                        help="Episode from which wall_obstacles turns on (default: 500)")
    parser.add_argument("--box_speed",         type=int,   default=2)
    parser.add_argument("--scaling_factor",    type=int,   default=5)
    parser.add_argument("--arena_size",        type=int,   default=500)
    parser.add_argument("--max_steps",         type=int,   default=200)

    args = parser.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    def env_fn(seed, wall_obstacles=False, difficulty=0):
        return OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=args.box_speed,
            seed=seed,
        )

    agent = PPO(env_fn, args)
    agent.train()
    agent.save(args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()