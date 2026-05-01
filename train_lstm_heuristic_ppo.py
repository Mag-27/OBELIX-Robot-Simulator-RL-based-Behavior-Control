from __future__ import annotations
import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
FW_IDX    = 2
TURN_IDXS = {0, 1, 3, 4}
N_OBS     = 18
N_OBS_AUG = 22
N_ACTIONS = 5
N_MODES   = 4

MODE_SEARCH  = 0
MODE_ALIGN   = 1
MODE_PUSH    = 2
MODE_UNSTUCK = 3

# =========================================================
# MODE HELPERS
# =========================================================

def get_mode(obs: np.ndarray) -> int:
    if obs[17]: return MODE_UNSTUCK
    if obs[16]: return MODE_PUSH
    if np.any(obs[0:16]): return MODE_ALIGN
    return MODE_SEARCH

def mode_to_onehot(mode: int) -> np.ndarray:
    v = np.zeros(N_MODES, dtype=np.float32)
    v[mode] = 1.0
    return v

def augment_obs(raw_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([raw_obs, mode_to_onehot(get_mode(raw_obs))])

def normalise_obs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)

# =========================================================
# SHAPED REWARD
# =========================================================

class RewardShaper:
    R_PROXIMITY     =  0.10
    R_ALIGN         =  0.08
    R_FW_SEARCH     =  0.02
    R_CONTACT_BONUS =  0.50
    R_SUSTAIN       =  0.08
    R_PUSH_STREAK   =  0.03
    R_LOST_CONTACT  = -0.20
    R_FLAT_ROT      = -0.01
    R_SPIN_STREAK   = -0.05
    R_STUCK_BASE    = -0.30  # strong — robot must not keep ramming
    R_STUCK_ESCAL   = -0.10  # per step beyond STUCK_GRACE
    R_GOAL          = 10.00
    R_TIMEOUT       = -1.00

    SPIN_GRACE  = 4
    STUCK_GRACE = 2

    SONAR_POINT_ANGLES = np.deg2rad([-90, -90, 0, 0, 0, 0, 90, 90])

    def reset(self):
        self._first_contact = False
        self._had_contact   = False
        self._ir_candidate  = False
        self._prev_ir       = 0.0
        self._consec_turns  = 0
        self._push_streak   = 0
        self._consec_stuck  = 0

    def step(self, obs: np.ndarray, action: int, env_reward: float,
             done: bool, truncated: bool) -> float:

        ir    = float(obs[16] > 0.5)
        stuck = float(obs[17] > 0.5)
        mode  = get_mode(obs)
        r     = 0.0

        # Confirm contact: IR fired last step + robot moved this step
        # Wall IR always causes immediate stuck so it never confirms
        if self._ir_candidate and not stuck:
            self._had_contact = True
        self._ir_candidate = bool(ir)

        # ---- proximity (SEARCH + ALIGN) ----
        if mode in (MODE_SEARCH, MODE_ALIGN):
            n = float(obs[0:16].sum())
            r += (n / 16.0) * self.R_PROXIMITY

        # ---- alignment (ALIGN only) ----
        if mode == MODE_ALIGN:
            unit_active = np.array([
                float(obs[2*i] > 0.5 or obs[2*i+1] > 0.5)
                for i in range(8)
            ])
            if unit_active.sum() > 0:
                cx = float(np.dot(unit_active, np.cos(self.SONAR_POINT_ANGLES)))
                cy = float(np.dot(unit_active, np.sin(self.SONAR_POINT_ANGLES)))
                theta_err = float(np.arctan2(cy, cx))
                alignment = 1.0 - abs(theta_err) / np.pi
                r += alignment * self.R_ALIGN

        # ---- forward bonus in SEARCH ----
        if mode == MODE_SEARCH and action == FW_IDX:
            r += self.R_FW_SEARCH

        # ---- first confirmed contact bonus ----
        if self._had_contact and not self._first_contact:
            r += self.R_CONTACT_BONUS
            self._first_contact = True

        # ---- sustain + push streak ----
        if self._had_contact and ir and not stuck:
            self._push_streak += 1
            r += self.R_SUSTAIN
            r += self.R_PUSH_STREAK * min(self._push_streak, 20)
        else:
            self._push_streak = max(0, self._push_streak - 1)

        # ---- lost contact ----
        if self._had_contact and self._prev_ir > 0.5 and ir < 0.5:
            r += self.R_LOST_CONTACT
        self._prev_ir = ir

        # ---- rotation penalties ----
        if action in TURN_IDXS:
            r += self.R_FLAT_ROT
            self._consec_turns += 1
        else:
            self._consec_turns = 0

        spin_excess = max(self._consec_turns - self.SPIN_GRACE, 0)
        if spin_excess > 0:
            r += self.R_SPIN_STREAK * spin_excess

        # ---- stuck: escalating, never positive ----
        if stuck:
            self._consec_stuck += 1
            r += self.R_STUCK_BASE
            excess = max(self._consec_stuck - self.STUCK_GRACE, 0)
            if excess > 0:
                r += self.R_STUCK_ESCAL * excess
        else:
            self._consec_stuck = 0

        # ---- terminal ----
        if done:
            if env_reward >= 1000:
                r += self.R_GOAL
            else:
                r += self.R_TIMEOUT

        return r


# =========================================================
# ENV LOADER
# =========================================================

def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def create_env(OBELIX, args, seed, wall_obstacles=False):
    return OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=seed,
    )

# =========================================================
# NETWORKS
# =========================================================

class PolicyNetwork(nn.Module):
    def __init__(self, stateDim=N_OBS_AUG, n_actions=N_ACTIONS,
                 hidden=128, temperature=1.5, prob_floor=0.03, fw_floor=0.05):
        super().__init__()
        self.temperature = temperature
        self.prob_floor  = prob_floor
        self.fw_floor    = fw_floor
        self.n_actions   = n_actions

        self.fc   = nn.Linear(stateDim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out  = nn.Linear(hidden, n_actions)
        # No FW bias at init — let reward shape the behavior
        nn.init.zeros_(self.out.bias)

    def forward(self, x, hidden=None, actions=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = torch.relu(self.fc(x))
        x, hidden = self.lstm(x, hidden)
        logits = self.out(x) / self.temperature
        probs  = torch.softmax(logits, dim=-1)
        probs  = (probs + self.prob_floor) / (1.0 + self.prob_floor * self.n_actions)

        # Small FW floor only — prevents complete FW suppression
        fw_col = probs[..., FW_IDX].clamp(min=self.fw_floor)
        mask   = torch.zeros_like(probs)
        mask[..., FW_IDX] = 1.0
        probs  = probs * (1 - mask) + fw_col.unsqueeze(-1) * mask
        probs  = probs / probs.sum(dim=-1, keepdim=True)

        dist = Categorical(probs=probs)
        a    = dist.sample() if actions is None else actions
        return a, dist.log_prob(a), dist.entropy(), probs.argmax(-1), hidden, probs


class ValueNetwork(nn.Module):
    def __init__(self, stateDim=N_OBS_AUG, hidden=128):
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
        self.shaper = RewardShaper()

        self.pnet = PolicyNetwork(
            temperature=args.temperature,
            prob_floor=args.prob_floor,
            fw_floor=args.fw_floor,
        )
        self.vnet = ValueNetwork()

        self.opt_p = optim.Adam(self.pnet.parameters(), lr=args.lr_policy)
        self.opt_v = optim.Adam(self.vnet.parameters(), lr=args.lr_value)

        self.total_steps = 0

    def _get_eps(self):
        return max(
            self.args.eps_end,
            self.args.eps_start - self.total_steps *
            (self.args.eps_start - self.args.eps_end) /
            max(1, self.args.eps_decay_steps)
        )

    def rollout(self, env, render=False):
        raw_s = np.asarray(env.reset(), dtype=np.float32)
        s     = normalise_obs(augment_obs(raw_s))
        done  = False

        self.shaper.reset()

        states, actions, rewards, logps, values = [], [], [], [], []
        h_p = h_v = None
        episode_success = False

        while not done:
            s_t = torch.tensor(s, dtype=torch.float32).view(1, 1, -1)

            with torch.no_grad():
                a_net, logp, _, _, h_p, _ = self.pnet(s_t, h_p)
                v, h_v                    = self.vnet(s_t, h_v)

            h_p = (h_p[0].detach(), h_p[1].detach())
            h_v = (h_v[0].detach(), h_v[1].detach())

            mode = get_mode(raw_s)
            eps  = self._get_eps()

            # Only nudge FW in SEARCH and only early in training
            if np.random.rand() < self.args.fw_rollout_rate and mode == MODE_SEARCH:
                a = FW_IDX
            elif np.random.rand() < eps:
                a = np.random.randint(N_ACTIONS)
            else:
                a = int(a_net.item())

            # In UNSTUCK mode: suppress FW, force a turn instead
            if mode == MODE_UNSTUCK and a == FW_IDX:
                a = np.random.choice([0, 1, 3, 4])

            self.total_steps += 1

            raw_s2, env_reward, done = env.step(ACTIONS[a], render=render)
            raw_s2 = np.asarray(raw_s2, dtype=np.float32)

            truncated = done and (env_reward < 1000)
            if env_reward >= 1000:
                episode_success = True

            shaped_r = self.shaper.step(raw_s2, a, float(env_reward), done, truncated)

            s2 = normalise_obs(augment_obs(raw_s2))

            states.append(s)
            actions.append(a)
            rewards.append(shaped_r)
            logps.append(float(logp.item()))
            values.append(float(v.item()))

            raw_s = raw_s2
            s     = s2

        return states, actions, rewards, logps, values, episode_success

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        adv  = []
        gae  = 0.0
        vals = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * vals[t+1] - vals[t]
            gae   = delta + gamma * lam * gae
            adv.insert(0, gae)
        returns = [a + v for a, v in zip(adv, values)]
        return adv, returns

    def train(self):
        seed = random.randint(0, 99999)

        for ep in range(self.args.total_episodes):

            use_walls = (ep // 10) % 2 == 1

            if ep % self.args.seed_every == 0:
                seed = random.randint(0, 99999)

            env       = self.env_fn(seed, wall_obstacles=use_walls)
            render_ep = (ep % self.args.eval_every == 0)

            states, actions, rewards, logps, values, success = self.rollout(
                env, render=render_ep
            )

            adv, returns = self.compute_gae(rewards, values)

            states_t  = torch.from_numpy(np.array(states, dtype=np.float32)).unsqueeze(1)
            actions_t = torch.tensor(actions, dtype=torch.long)
            old_logps = torch.tensor(logps,   dtype=torch.float32)
            adv_t     = torch.tensor(adv,     dtype=torch.float32)
            returns_t = torch.tensor(returns, dtype=torch.float32)

            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            adv_t = torch.clamp(adv_t, -3.0, 5.0)

            entropy_coef = max(0.01, 0.05 * (1.0 - ep / self.args.total_episodes))

            for _ in range(self.args.ppo_epochs):
                _, logp, ent, _, _, _ = self.pnet(
                    states_t, actions=actions_t.unsqueeze(1))
                logp = logp.squeeze(1)

                ratio = torch.exp(logp - old_logps)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t

                loss_p = -torch.min(surr1, surr2).mean() - entropy_coef * ent.mean()

                v_pred, _ = self.vnet(states_t)
                loss_v    = ((v_pred.squeeze() - returns_t) ** 2).mean()

                self.opt_p.zero_grad()
                loss_p.backward()
                nn.utils.clip_grad_norm_(self.pnet.parameters(), 0.5)
                self.opt_p.step()

                self.opt_v.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.vnet.parameters(), 0.5)
                self.opt_v.step()

            with torch.no_grad():
                _, _, _, _, _, probs_dbg = self.pnet(
                    states_t[:8], actions=actions_t[:8].unsqueeze(1))
                mean_p = probs_dbg.squeeze(1).mean(0)

            phase   = "walls" if use_walls else "     "
            act_str = "  ".join(f"{ACTIONS[i]}:{mean_p[i]*100:.1f}%" for i in range(N_ACTIONS))
            flag    = "  SUCCESS" if success else ""
            fw_warn = "  ◀ FW LOW!" if mean_p[FW_IDX] < 0.08 else ""
            stuck_steps = sum(1 for r in rewards if r <= self.shaper.R_STUCK_BASE)
            print(f"Ep {ep:4d} | {phase} | ret={sum(rewards):+8.2f} | "
                  f"stuck={stuck_steps:3d} | eps={self._get_eps():.3f} | "
                  f"[{act_str}]{flag}{fw_warn}")

        self.save(self.args.out)
        print(f"\nSaved → {self.args.out}")

    def save(self, path):
        torch.save(self.pnet.state_dict(), path)


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--obelix_py",       type=str,   required=True)
    parser.add_argument("--out",             type=str,   default="weights_ppo.pth")
    parser.add_argument("--total_episodes",  type=int,   default=500)
    parser.add_argument("--eval_every",      type=int,   default=20)
    parser.add_argument("--seed_every",      type=int,   default=5)
    parser.add_argument("--ppo_epochs",      type=int,   default=8)

    parser.add_argument("--temperature",     type=float, default=1.5)
    parser.add_argument("--prob_floor",      type=float, default=0.03)
    parser.add_argument("--fw_floor",        type=float, default=0.05)   # was 0.15
    parser.add_argument("--fw_rollout_rate", type=float, default=0.10)   # was 0.20
    parser.add_argument("--eps_start",       type=float, default=0.20)
    parser.add_argument("--eps_end",         type=float, default=0.02)
    parser.add_argument("--eps_decay_steps", type=int,   default=60000)

    parser.add_argument("--lr_policy",       type=float, default=1e-4)
    parser.add_argument("--lr_value",        type=float, default=1e-3)
    parser.add_argument("--difficulty",      type=int,   default=0)
    parser.add_argument("--box_speed",       type=int,   default=2)
    parser.add_argument("--scaling_factor",  type=int,   default=5)
    parser.add_argument("--arena_size",      type=int,   default=500)
    parser.add_argument("--max_steps",       type=int,   default=200)

    args = parser.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    def env_fn(seed, wall_obstacles=False):
        return create_env(OBELIX, args, seed, wall_obstacles=wall_obstacles)

    PPO(env_fn, args).train()

if __name__ == "__main__":
    main()