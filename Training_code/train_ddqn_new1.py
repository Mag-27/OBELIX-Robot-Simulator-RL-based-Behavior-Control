"""
DDQN Trainer for OBELIX
- Mode-aware reward shaper (no hardcoded action overrides)
- Renders every episode
- Curriculum: alternates 10 eps no-wall / 10 eps wall
- Plots all relevant training variables after training
"""

from __future__ import annotations

import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
FW_IDX    = 2
TURN_IDXS = {0, 1, 3, 4}
N_OBS     = 18
N_ACTIONS = 5

DEVICE = torch.device("cpu")
print("[trainer] device:", DEVICE)

# Sonar unit angles: 8 units, bit 2i=far, 2i+1=near
SONAR_ANGLES = np.deg2rad([-90, -90, 0, 0, 0, 0, 90, 90])


# ---------------------------------------------------------------------------
# OBELIX helpers
# ---------------------------------------------------------------------------

def import_obelix(path):
    import importlib.util
    spec   = importlib.util.spec_from_file_location("obelix_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OBELIX

def env_reset(env):
    return np.asarray(env.reset(), dtype=np.float32)

def env_step(env, action, render=False):
    obs, r, done = env.step(action, render=render)
    return np.asarray(obs, dtype=np.float32), float(r), bool(done)


# ---------------------------------------------------------------------------
# Mode detection  (used ONLY by reward shaper — never seen by the agent)
# ---------------------------------------------------------------------------

def get_mode(obs):
    """
    0=SEARCH  nothing visible
    1=ALIGN   sonar bits on, no IR
    2=PUSH    IR contact
    3=UNSTUCK stuck flag
    """
    if obs[17] > 0.5: return 3
    if obs[16] > 0.5: return 2
    if np.any(obs[0:16] > 0.5): return 1
    return 0


# ---------------------------------------------------------------------------
# Reward Shaper
# ---------------------------------------------------------------------------

class RewardShaper:

    # SEARCH
    R_SCAN_TURN    =  0.12
    R_BLIND_FW     = -0.50
    R_FIRST_DETECT =  0.40

    # ALIGN
    R_CENTRING     =  0.12
    R_FW_ALIGNED   =  0.10
    R_FW_MISALIGN  = -0.06
    R_SPIN_ALIGN   = -0.05
    SPIN_GRACE     =  3

    # PUSH / CONTACT
    R_CONTACT      =  0.60
    R_SUSTAIN      =  0.08
    R_STREAK       =  0.02
    R_LOST         = -0.20

    # UNSTUCK
    R_STUCK_BASE   = -1.00
    R_STUCK_ESCAL  = -0.10
    STUCK_GRACE    =  2
    R_TURN_STUCK   =  0.05
    R_FW_STUCK     = -0.50

    # Terminal
    R_GOAL         = 10.00
    R_TIMEOUT      = -0.50

    def reset(self):
        self._first_detect       = False
        self._first_contact      = False
        self._had_contact        = False
        self._prev_ir            = 0.0
        self._push_streak        = 0
        self._consec_stuck       = 0
        self._consec_turns_align = 0

    def step(self, obs_prev, obs_next, action, env_reward, done):
        mode        = get_mode(obs_prev)
        stuck_after = float(obs_next[17] > 0.5)
        ir_before   = float(obs_prev[16] > 0.5)
        ir_after    = float(obs_next[16] > 0.5)
        is_fw       = (action == FW_IDX)
        is_turn     = (action in TURN_IDXS)
        r           = 0.0

        # -- SEARCH --
        if mode == 0:
            if is_turn:
                r += self.R_SCAN_TURN
            if is_fw:
                r += self.R_BLIND_FW
            if not self._first_detect and np.any(obs_next[0:16] > 0.5):
                self._first_detect = True
                r += self.R_FIRST_DETECT

        # -- ALIGN --
        if mode == 1:
            unit_active = np.array([
                float(obs_prev[2*i] > 0.5 or obs_prev[2*i+1] > 0.5)
                for i in range(8)
            ])
            if unit_active.sum() > 0:
                cx        = float(np.dot(unit_active, np.cos(SONAR_ANGLES)))
                cy        = float(np.dot(unit_active, np.sin(SONAR_ANGLES)))
                theta_err = float(np.arctan2(cy, cx))
                alignment = 1.0 - abs(theta_err) / np.pi
                r += alignment * self.R_CENTRING
                if is_fw:
                    r += self.R_FW_ALIGNED if abs(theta_err) < np.pi / 4 \
                         else self.R_FW_MISALIGN
                if is_turn:
                    self._consec_turns_align += 1
                    excess = max(0, self._consec_turns_align - self.SPIN_GRACE)
                    if excess > 0:
                        r += self.R_SPIN_ALIGN * excess
                else:
                    self._consec_turns_align = 0
            else:
                self._consec_turns_align = 0

        # -- CONTACT / PUSH --
        if ir_after and not self._had_contact and not stuck_after:
            self._had_contact = True
        if self._had_contact and not self._first_contact:
            r += self.R_CONTACT
            self._first_contact = True

        if mode == 2:
            if ir_after and not stuck_after:
                self._push_streak += 1
                r += self.R_SUSTAIN
                r += self.R_STREAK * min(self._push_streak, 20)
            else:
                self._push_streak = max(0, self._push_streak - 1)

        if self._had_contact and ir_before > 0.5 and ir_after < 0.5:
            r += self.R_LOST
        self._prev_ir = ir_after

        # -- UNSTUCK --
        if stuck_after:
            self._consec_stuck += 1
            r += self.R_STUCK_BASE
            excess = max(0, self._consec_stuck - self.STUCK_GRACE)
            if excess > 0:
                r += self.R_STUCK_ESCAL * excess
            if is_turn:
                r += self.R_TURN_STUCK
            if is_fw:
                r += self.R_FW_STUCK
        else:
            self._consec_stuck = 0

        # -- Terminal --
        if done:
            r += self.R_GOAL if env_reward >= 1000 else self.R_TIMEOUT

        return r


# ---------------------------------------------------------------------------
# Q Network
# ---------------------------------------------------------------------------

def createValueNetwork():
    return nn.Sequential(
        nn.Linear(N_OBS, 128), nn.ReLU(),
        nn.Linear(128, 128),   nn.ReLU(),
        nn.Linear(128, N_ACTIONS)
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch):
        idx        = np.random.choice(len(self.buffer), batch, replace=False)
        batch_data = [self.buffer[i] for i in idx]
        s, a, r, ns, d = zip(*batch_data)
        return (np.array(s), np.array(a), np.array(r),
                np.array(ns), np.array(d))

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

class EpsilonGreedy:

    def __init__(self, start=1.0, end=0.05, decay_steps=900000):
        self.epsilon = start
        self.end     = end
        self.decay   = (start - end) / decay_steps

    def select(self, net, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        s = torch.tensor(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return int(net(s).argmax().item())

    def step(self):
        self.epsilon = max(self.end, self.epsilon - self.decay)


class GreedyPolicy:

    def select(self, net, state):
        s = torch.tensor(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return int(net(s).argmax().item())


# ---------------------------------------------------------------------------
# DDQN Agent
# ---------------------------------------------------------------------------

class DDQN:

    def __init__(self, env_fn, gamma, buffer_size, batch_size, lr, update_freq):
        self.env_fn      = env_fn
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.update_freq = update_freq

        self.online = createValueNetwork()
        self.target = createValueNetwork()
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer   = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer      = ReplayBuffer(buffer_size)
        self.total_steps = 0

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s  = torch.tensor(s).float().to(DEVICE)
        a  = torch.tensor(a).long().to(DEVICE)
        r  = torch.tensor(r).float().to(DEVICE)
        ns = torch.tensor(ns).float().to(DEVICE)
        d  = torch.tensor(d).float().to(DEVICE)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a = self.online(ns).argmax(1)
            next_q = self.target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * next_q * (1 - d)

        loss = ((target - q) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(agent, env_fn, epochs, episodes_per_epoch, max_steps,
          curriculum_block=10):
    explorer  = EpsilonGreedy()
    evaluator = GreedyPolicy()
    shaper    = RewardShaper()

    history = {
        "train_raw":    [],
        "train_shaped": [],
        "eval_raw":     [],
        "eval_shaped":  [],
        "epsilon":      [],
        "loss":         [],
        "stuck_frac":   [],
        "fw_frac":      [],
        "success":      [],
        "wall_flag":    [],
        "ep_length":    [],
    }

    ep_global = 0

    for epoch in range(epochs):

        epoch_losses = []
        history["epsilon"].append(explorer.epsilon)

        for ep in range(episodes_per_epoch):

            block_idx = (ep_global // curriculum_block) % 2
            use_walls = (block_idx == 1)

            env   = env_fn(seed=epoch * 1000 + ep, wall_obstacles=use_walls)
            state = env_reset(env)
            shaper.reset()

            total_raw    = 0.0
            total_shaped = 0.0
            ep_losses    = []
            stuck_steps  = 0
            fw_steps     = 0
            success      = False
            ep_steps     = 0

            for step in range(max_steps):

                action = explorer.select(agent.online, state)

                next_state, env_r, done = env_step(
                    env, ACTIONS[action], render=False)

                shaped_r = shaper.step(
                    obs_prev=state,
                    obs_next=next_state,
                    action=action,
                    env_reward=env_r,
                    done=done,
                )

                agent.buffer.add(state, action, shaped_r, next_state, done)

                total_raw    += env_r
                total_shaped += shaped_r
                stuck_steps  += int(next_state[17] > 0.5)
                fw_steps     += int(action == FW_IDX)
                ep_steps     += 1

                if env_r >= 1000:
                    success = True

                state              = next_state
                agent.total_steps += 1
                explorer.step()

                loss_val = agent.train_step()
                if loss_val is not None:
                    ep_losses.append(loss_val)

                if agent.total_steps % agent.update_freq == 0:
                    agent.update_target()

                if done:
                    break

            history["train_raw"].append(total_raw)
            history["train_shaped"].append(total_shaped)
            history["stuck_frac"].append(stuck_steps / max(ep_steps, 1))
            history["fw_frac"].append(fw_steps / max(ep_steps, 1))
            history["success"].append(int(success))
            history["wall_flag"].append(int(use_walls))
            history["ep_length"].append(ep_steps)
            history["loss"].append(
                float(np.mean(ep_losses)) if ep_losses else 0.0)
            epoch_losses.extend(ep_losses)

            ep_global += 1

        # Evaluation (greedy)
        eval_raw_ep    = []
        eval_shaped_ep = []
        for i in range(3):
            env   = env_fn(seed=90000 + i, wall_obstacles=True if i % 2 == 0 else False)
            state = env_reset(env)
            shaper.reset()
            ep_raw = 0.0
            ep_sh  = 0.0
            for _ in range(max_steps):
                action                  = evaluator.select(agent.online, state)
                next_state, env_r, done = env_step(env, ACTIONS[action])
                ep_sh  += shaper.step(state, next_state, action, env_r, done)
                ep_raw += env_r
                state   = next_state
                if done:
                    break
            eval_raw_ep.append(ep_raw)
            eval_shaped_ep.append(ep_sh)

        history["eval_raw"].append(float(np.mean(eval_raw_ep)))
        history["eval_shaped"].append(float(np.mean(eval_shaped_ep)))

        n = episodes_per_epoch
        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"TrainRaw {np.mean(history['train_raw'][-n:]):+8.1f} | "
            f"Shaped {np.mean(history['train_shaped'][-n:]):+8.2f} | "
            f"EvalRaw {history['eval_raw'][-1]:+8.1f} | "
            f"Eps {explorer.epsilon:.3f} | "
            f"Succ {sum(history['success'][-n:])}/{n} | "
            f"Loss {np.mean(epoch_losses) if epoch_losses else 0:.4f}"
        )

    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(x, w=20):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(history, out_path="training_plots.png"):

    n   = len(history["train_raw"])
    x   = np.arange(n)
    wm  = np.array(history["wall_flag"], dtype=bool)
    nwm = ~wm

    BG   = "#ffffff"
    PANEL= "#f8f8f8"
    GRID = "#e0e0e0"
    C1   = "#38d9a9"
    C2   = "#ff6b6b"
    C3   = "#69db7c"
    C4   = "#ffd43b"
    C5   = "#a78bfa"
    ALF  = 0.12

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    def ax_style(ax, title, xlabel="Episode", ylabel=""):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color="#111111", fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color="#555555", fontsize=8)
        ax.set_ylabel(ylabel, color="#555555", fontsize=8)
        ax.tick_params(colors="#333333")
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--")

    def line(ax, y, color, label=None):
        arr = np.array(y)
        ax.plot(x, arr, color="#aaaaaa", lw=0.4, alpha=0.5)
        sm  = smooth(arr)
        if len(sm):
            ax.plot(np.arange(len(sm)), sm, color=color, lw=1.5, label=label)

    # ── 1. Raw episode return ─────────────────────────────────────────────────
    ax  = fig.add_subplot(gs[0, 0])
    raw = np.array(history["train_raw"])
    ax.fill_between(x[wm],  raw[wm],  alpha=ALF, color=C2)
    ax.fill_between(x[nwm], raw[nwm], alpha=ALF, color=C1)
    ax.plot(x, raw, color="#aaaaaa", lw=0.4, alpha=0.5)
    sm = smooth(raw)
    if len(sm):
        ax.plot(np.arange(len(sm)), sm, color=C1, lw=1.8, label="smoothed")
    ax.axhline(0, color="#aaaaaa", lw=0.5, ls="--")
    ax.plot([], [], color=C2, lw=6, alpha=ALF, label="wall on")
    ax.plot([], [], color=C1, lw=6, alpha=ALF, label="no wall")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#111111", framealpha=0.6)
    ax_style(ax, "Raw Episode Return", ylabel="Return")

    # ── 2. Shaped episode return ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    line(ax, history["train_shaped"], C4)
    ax.axhline(0, color="#aaaaaa", lw=0.5, ls="--")
    ax_style(ax, "Shaped Episode Return", ylabel="Shaped Return")

    # ── 3. Eval raw return per epoch ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ev = np.array(history["eval_raw"])
    ax.plot(np.arange(len(ev)), ev, color=C2, lw=1.5,
            marker="o", markersize=3)
    ax.axhline(0, color="#aaaaaa", lw=0.5, ls="--")
    ax_style(ax, "Eval Raw Return (per epoch)",
             xlabel="Epoch", ylabel="Return")

    # ── 4. Success rate ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    su = np.array(history["success"], dtype=float)
    ax.bar(x[wm],  su[wm],  color=C2, alpha=0.7, width=1.0, label="wall")
    ax.bar(x[nwm], su[nwm], color=C3, alpha=0.7, width=1.0, label="no wall")
    sm = smooth(su)
    if len(sm):
        ax.plot(np.arange(len(sm)), sm, color="#333333", lw=1.5, label="rolling")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#111111", framealpha=0.6)
    ax_style(ax, "Episode Success", ylabel="Success (0/1)")

    # ── 5. Epsilon decay ──────────────────────────────────────────────────────
    ax     = fig.add_subplot(gs[1, 1])
    ep_arr = np.array(history["epsilon"])
    ax.plot(np.arange(len(ep_arr)), ep_arr, color=C5, lw=1.8)
    ax.set_ylim(0, 1.05)
    ax_style(ax, "Epsilon Decay", xlabel="Epoch", ylabel="ε")

    # ── 6. TD loss ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    line(ax, history["loss"], C5)
    ax_style(ax, "TD Loss per Episode", ylabel="MSE Loss")

    # ── 7. Stuck fraction ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    line(ax, history["stuck_frac"], C2)
    ax.set_ylim(0, 1.0)
    ax_style(ax, "Stuck Fraction per Episode", ylabel="Fraction")

    # ── 8. FW action fraction ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    line(ax, history["fw_frac"], C4)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.20, color=C4, lw=0.7, ls="--", alpha=0.5,
               label="20% reference")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#111111", framealpha=0.6)
    ax_style(ax, "FW Action Fraction per Episode", ylabel="Fraction")

    # ── 9. Episode length ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    line(ax, history["ep_length"], C1)
    ax_style(ax, "Episode Length", ylabel="Steps")

    fig.suptitle("DDQN-OBELIX Training Dashboard",
                 color="#111111", fontsize=15, y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved → {out_path}")

    # --- Save individual plots ---
    base = out_path.replace(".png", "")
    individual_data = [
        ("raw_return",     x, np.array(history["train_raw"]),    C1, "Raw Episode Return",    "Return"),
        ("shaped_return",  x, np.array(history["train_shaped"]), C4, "Shaped Episode Return", "Return"),
        ("stuck_fraction", x, np.array(history["stuck_frac"]),   C2, "Stuck Fraction",        "Fraction"),
        ("fw_fraction",    x, np.array(history["fw_frac"]),      C4, "FW Action Fraction",    "Fraction"),
        ("episode_length", x, np.array(history["ep_length"]),    C1, "Episode Length",        "Steps"),
        ("loss",           x, np.array(history["loss"]),         C5, "TD Loss",               "MSE Loss"),
    ]
    for name, xi, yi, color, title, ylabel in individual_data:
        fig_i, ax_i = plt.subplots(figsize=(7, 4))
        fig_i.patch.set_facecolor(BG)
        ax_i.set_facecolor(PANEL)
        arr = np.array(yi, dtype=float)
        ax_i.plot(xi, arr, color=color, lw=0.3, alpha=0.2)
        sm = smooth(arr)
        if len(sm):
            ax_i.plot(np.arange(len(sm)), sm, color=color, lw=1.8)
        ax_style(ax_i, title, ylabel=ylabel)
        plt.tight_layout()
        p = f"{base}_ddqn_{name}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig_i.get_facecolor())
        plt.close()
        print(f"[plot] saved → {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--obelix_py",          required=True)
    parser.add_argument("--out",                default="ddqn_weights.pth")
    parser.add_argument("--plot_out",           default="training_plots.png")
    parser.add_argument("--epochs",             type=int,   default=1)
    parser.add_argument("--episodes_per_epoch", type=int,   default=40)
    parser.add_argument("--max_steps",          type=int,   default=300)
    parser.add_argument("--curriculum_block",   type=int,   default=10,
                        help="Episodes per wall/no-wall block (default 10)")
    parser.add_argument("--scaling_factor",     type=int,   default=5)
    parser.add_argument("--arena_size",         type=int,   default=500)
    parser.add_argument("--difficulty",         type=int,   default=0)
    parser.add_argument("--box_speed",          type=int,   default=2)

    args = parser.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    def env_fn(seed, wall_obstacles=False):
        return OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=seed,
        )

    agent = DDQN(
        env_fn,
        gamma=0.98,
        buffer_size=50000,
        batch_size=64,
        lr=1e-4,
        update_freq=1000,
    )

    history = train(
        agent, env_fn,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        max_steps=args.max_steps,
        curriculum_block=args.curriculum_block,
    )

    torch.save(agent.online.state_dict(), args.out)
    print(f"\n[trainer] weights saved → {args.out}")

    plot_training(history, out_path=args.plot_out)
    print("[trainer] done.")


if __name__ == "__main__":
    main()