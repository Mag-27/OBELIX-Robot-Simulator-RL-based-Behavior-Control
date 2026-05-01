from __future__ import annotations
import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import deque

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
MODE_NAMES   = ["SEARCH", "ALIGN", "PUSH", "UNSTUCK"]

SONAR_POINT_ANGLES = np.deg2rad([-90, -90, 0, 0, 0, 0, 90, 90])

# =========================================================
# PALETTE — light theme
# =========================================================
BG       = "#ffffff"
PANEL    = "#f8f8f8"
GRID_C   = "#e0e0e0"
WHITE    = "#111111"
DIM      = "#555555"

C_RETURN  = "#38d9a9"   # teal  — returns
C_SHAPED  = "#ffd43b"   # amber — shaped
C_LOSS    = "#ff6b6b"   # coral — loss
C_EPS     = "#a78bfa"   # violet — epsilon
C_FW      = "#74c0fc"   # sky   — forward action
C_STUCK   = "#ff922b"   # orange — stuck
C_SUCCESS = "#69db7c"   # green — success
C_WALL    = "#e64980"   # pink  — wall episodes
C_NOWALL  = "#339af0"   # blue  — no-wall episodes

# Mode colours
MODE_COLORS = ["#339af0", "#51cf66", "#ff6b6b", "#ff922b"]   # S A P U

# =========================================================
# STUCK ESCAPE FSM
# =========================================================

class StuckEscapeFSM:
    TURN_STEPS     = 4
    COOLDOWN_STEPS = 5
    MAX_ATTEMPTS   = 6

    def reset(self):
        self._state    = "IDLE"
        self._count    = 0
        self._dir      = 0
        self._attempts = 0

    def step(self, stuck_now: bool) -> int | None:
        if self._state == "IDLE":
            if stuck_now:
                self._state    = "TURNING"
                self._count    = 0
                self._attempts += 1
                self._dir      = 0 if self._attempts % 2 == 1 else 4
            return None
        if self._state == "TURNING":
            self._count += 1
            action = self._dir
            if self._count >= self.TURN_STEPS:
                self._state = "PROBE"
                self._count = 0
            return action
        if self._state == "PROBE":
            if not stuck_now:
                self._state    = "COOLDOWN"
                self._count    = 0
                self._attempts = 0
            else:
                self._state    = "TURNING"
                self._count    = 0
                self._dir      = 4 if self._dir == 0 else 0
                if self._attempts >= self.MAX_ATTEMPTS:
                    self._attempts = 0
                    return random.choice([0, 1, 3, 4])
            return FW_IDX
        if self._state == "COOLDOWN":
            self._count += 1
            if self._count >= self.COOLDOWN_STEPS:
                self._state = "IDLE"
            return None
        return None

    @property
    def is_active(self) -> bool:
        return self._state in ("TURNING", "PROBE")


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
# REWARD SHAPER
# =========================================================

class RewardShaper:
    R_SCAN_TURN      =  0.12
    R_BLIND_FW       = -0.50
    R_FIRST_DETECT   =  0.30
    R_CENTRING       =  0.10
    R_FW_ALIGNED     =  0.08
    R_FW_MISALIGNED  = -0.04
    R_SPIN_ALIGN     = -0.05
    SPIN_GRACE_ALIGN =  3
    R_CONTACT_BONUS  =  0.50
    R_SUSTAIN        =  0.06
    R_PUSH_STREAK    =  0.02
    R_LOST_CONTACT   = -0.15
    R_UTURN_STUCK    =  0.04
    R_FW_STUCK       = -0.40
    R_STUCK_BASE     = -1.00
    R_STUCK_ESCAL    = -0.08
    STUCK_GRACE      =  2
    R_GOAL           = 10.00
    R_TIMEOUT        = -0.50

    def reset(self):
        self._first_contact      = False
        self._had_contact        = False
        self._first_detect       = False
        self._prev_ir            = 0.0
        self._consec_turns_align = 0
        self._push_streak        = 0
        self._consec_stuck       = 0

    def step(self, obs_prev, obs_next, action, env_reward,
             done, truncated, fsm_was_active) -> float:
        stuck_after = float(obs_next[17] > 0.5)
        ir_before   = float(obs_prev[16] > 0.5)
        ir_after    = float(obs_next[16] > 0.5)
        mode        = get_mode(obs_prev)
        is_fw       = (action == FW_IDX)
        is_turn     = (action in TURN_IDXS)
        r           = 0.0

        if mode == MODE_SEARCH:
            if is_turn: r += self.R_SCAN_TURN
            if is_fw:   r += self.R_BLIND_FW
            if not self._first_detect and np.any(obs_next[0:16] > 0.5):
                self._first_detect = True
                r += self.R_FIRST_DETECT

        if mode == MODE_ALIGN:
            unit_active = np.array([
                float(obs_prev[2*i] > 0.5 or obs_prev[2*i+1] > 0.5)
                for i in range(8)
            ])
            if unit_active.sum() > 0:
                cx        = float(np.dot(unit_active, np.cos(SONAR_POINT_ANGLES)))
                cy        = float(np.dot(unit_active, np.sin(SONAR_POINT_ANGLES)))
                theta_err = float(np.arctan2(cy, cx))
                alignment = 1.0 - abs(theta_err) / np.pi
                r += alignment * self.R_CENTRING
                if is_fw:
                    r += self.R_FW_ALIGNED if abs(theta_err) < np.pi/4 \
                         else self.R_FW_MISALIGNED
                if is_turn:
                    self._consec_turns_align += 1
                    excess = max(self._consec_turns_align - self.SPIN_GRACE_ALIGN, 0)
                    if excess > 0: r += self.R_SPIN_ALIGN * excess
                else:
                    self._consec_turns_align = 0
            else:
                self._consec_turns_align = 0

        if ir_after and not self._had_contact and not stuck_after:
            self._had_contact = True
        if self._had_contact and not self._first_contact:
            r += self.R_CONTACT_BONUS
            self._first_contact = True

        if mode == MODE_PUSH:
            if ir_after and not stuck_after:
                self._push_streak += 1
                r += self.R_SUSTAIN
                r += self.R_PUSH_STREAK * min(self._push_streak, 20)
            else:
                self._push_streak = max(0, self._push_streak - 1)

        if self._had_contact and ir_before > 0.5 and ir_after < 0.5:
            r += self.R_LOST_CONTACT
        self._prev_ir = ir_after

        if stuck_after:
            self._consec_stuck += 1
            r += self.R_STUCK_BASE
            excess = max(self._consec_stuck - self.STUCK_GRACE, 0)
            if excess > 0: r += self.R_STUCK_ESCAL * excess
            if is_fw and not fsm_was_active: r += self.R_FW_STUCK
            if is_turn: r += self.R_UTURN_STUCK
        else:
            self._consec_stuck = 0

        if done:
            r += self.R_GOAL if env_reward >= 1000 else self.R_TIMEOUT

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
                 hidden=128, temperature=1.5, prob_floor=0.03, fw_floor=0.0):
        super().__init__()
        self.temperature = temperature
        self.prob_floor  = prob_floor
        self.fw_floor    = fw_floor
        self.n_actions   = n_actions
        self.fc   = nn.Linear(stateDim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out  = nn.Linear(hidden, n_actions)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, hidden=None, actions=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = torch.relu(self.fc(x))
        x, hidden = self.lstm(x, hidden)
        logits = self.out(x) / self.temperature
        probs  = torch.softmax(logits, dim=-1)
        probs  = (probs + self.prob_floor) / (1.0 + self.prob_floor * self.n_actions)
        if self.fw_floor > 0:
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
# PLOTTING UTILITIES
# =========================================================

def _smooth(arr, w=15):
    if len(arr) < w:
        return np.array(arr, dtype=float)
    kernel = np.exp(-0.5 * np.linspace(-2, 2, w) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _ax_base(ax, title, xlabel="Episode", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color="#111111", fontsize=9, pad=5, fontweight="bold")
    ax.set_xlabel(xlabel, color="#555555", fontsize=7.5)
    ax.set_ylabel(ylabel, color="#555555", fontsize=7.5)
    ax.tick_params(colors="#333333", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_C)
    ax.grid(color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.7)


def _raw_and_smooth(ax, x, y, color, alpha_raw=0.18, lw=1.8, label=None):
    arr = np.array(y, dtype=float)
    ax.plot(x, arr, color=color, lw=0.3, alpha=alpha_raw)
    sm = _smooth(arr)
    ax.plot(x, sm, color=color, lw=lw, label=label)


def plot_training(history: dict, out_path: str = "training_plots.png"):
    n    = len(history["ep_returns"])
    x    = np.arange(n)
    wall = np.array(history["ep_wall"], dtype=bool)

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(BG)
    fig.suptitle("PPO-LSTM · OBELIX · Training Dashboard",
                 color="#111111", fontsize=14, fontweight="bold", y=0.975)
    gs = gridspec.GridSpec(
        5, 4, figure=fig,
        hspace=0.62, wspace=0.38,
        left=0.06, right=0.97, top=0.95, bottom=0.04,
    )

    # ── Row 0: Returns ────────────────────────────────────────────────────────

    ax = fig.add_subplot(gs[0, 0:2])
    ret = np.array(history["ep_returns"], dtype=float)
    for i in range(n):
        ax.axvspan(i - 0.5, i + 0.5,
                   color=C_WALL if wall[i] else C_NOWALL,
                   alpha=0.07, linewidth=0)
    ax.axhline(0, color=DIM, lw=0.5, ls="--")
    _raw_and_smooth(ax, x, ret, C_RETURN, label="shaped return")
    ax.plot([], [], color=C_WALL,   lw=6, alpha=0.25, label="wall on")
    ax.plot([], [], color=C_NOWALL, lw=6, alpha=0.25, label="no wall")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE,
              framealpha=0.6, loc="upper left")
    _ax_base(ax, "Shaped Episode Return", ylabel="Return")

    ax = fig.add_subplot(gs[0, 2:4])
    raw = np.array(history["ep_returns_raw"], dtype=float)
    for i in range(n):
        ax.axvspan(i - 0.5, i + 0.5,
                   color=C_WALL if wall[i] else C_NOWALL,
                   alpha=0.07, linewidth=0)
    ax.axhline(0, color=DIM, lw=0.5, ls="--")
    _raw_and_smooth(ax, x, raw, C_SHAPED, label="env return")
    succ = np.array(history["ep_success"], dtype=bool)
    if succ.any():
        ax.scatter(x[succ], raw[succ], color=C_SUCCESS,
                   s=18, zorder=5, label="success", marker="*")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.6)
    _ax_base(ax, "Raw Environment Return + Successes", ylabel="Return")

    # ── Row 1: Mode distribution ──────────────────────────────────────────────

    ax = fig.add_subplot(gs[1, 0:4])
    mf = np.array(history["ep_mode_fracs"], dtype=float)
    row_sum = mf.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    mf = mf / row_sum
    smooth_mf = np.column_stack([_smooth(mf[:, m]) for m in range(N_MODES)])
    smooth_mf = np.clip(smooth_mf, 0, None)
    s_sum = smooth_mf.sum(axis=1, keepdims=True)
    s_sum[s_sum == 0] = 1
    smooth_mf = smooth_mf / s_sum

    bottom = np.zeros(n)
    for m in range(N_MODES):
        ax.fill_between(x, bottom, bottom + smooth_mf[:, m],
                        color=MODE_COLORS[m], alpha=0.75, label=MODE_NAMES[m])
        bottom += smooth_mf[:, m]

    switches = np.where(np.diff(wall.astype(int)) != 0)[0]
    for sw in switches:
        ax.axvline(sw, color=WHITE, lw=0.6, ls=":", alpha=0.5)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE, framealpha=0.7,
              loc="upper right", ncol=4)
    _ax_base(ax, "Heuristic Mode Distribution per Episode (smoothed)",
             ylabel="Fraction of Steps")

    # ── Row 2: Action fractions ───────────────────────────────────────────────

    ACT_COLORS = ["#f03e3e", "#fd7e14", C_FW, "#20c997", "#845ef7"]

    ax = fig.add_subplot(gs[2, 0:2])
    af = np.array(history["ep_action_fracs"], dtype=float)
    af_s = np.column_stack([_smooth(af[:, a]) for a in range(N_ACTIONS)])
    af_s = np.clip(af_s, 0, None)
    a_sum = af_s.sum(axis=1, keepdims=True)
    a_sum[a_sum == 0] = 1
    af_s = af_s / a_sum
    bottom = np.zeros(n)
    for i, act in enumerate(ACTIONS):
        ax.fill_between(x, bottom, bottom + af_s[:, i],
                        color=ACT_COLORS[i], alpha=0.75, label=act)
        bottom += af_s[:, i]
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE,
              framealpha=0.7, ncol=5)
    _ax_base(ax, "Action Distribution (smoothed)", ylabel="Fraction")

    ax = fig.add_subplot(gs[2, 2])
    fw = np.array(history["ep_fw_frac"], dtype=float)
    _raw_and_smooth(ax, x, fw, C_FW)
    ax.axhline(0.20, color=C_FW, lw=0.7, ls="--", alpha=0.5, label="20% ref")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.6)
    _ax_base(ax, "FW Action Fraction", ylabel="Fraction")

    ax = fig.add_subplot(gs[2, 3])
    fsm = np.array(history["ep_fsm_frac"], dtype=float)
    _raw_and_smooth(ax, x, fsm, C_WALL)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _ax_base(ax, "FSM Active Fraction (stuck-escape)", ylabel="Fraction")

    # ── Row 3: Stuck, entropy, losses ─────────────────────────────────────────

    ax = fig.add_subplot(gs[3, 0])
    sk = np.array(history["ep_stuck_frac"], dtype=float)
    _raw_and_smooth(ax, x, sk, C_STUCK)
    ax.set_ylim(0, max(sk.max() * 1.1, 0.05))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _ax_base(ax, "Stuck Fraction per Episode", ylabel="Fraction")

    ax = fig.add_subplot(gs[3, 1])
    el = np.array(history["ep_lengths"], dtype=float)
    _raw_and_smooth(ax, x, el, C_RETURN)
    _ax_base(ax, "Episode Length", ylabel="Steps")

    ax = fig.add_subplot(gs[3, 2])
    ent = np.array(history["ep_entropy"], dtype=float)
    _raw_and_smooth(ax, x, ent, C_EPS)
    ax.axhline(np.log(N_ACTIONS), color=C_EPS, lw=0.6, ls="--",
               alpha=0.5, label="max entropy")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.6)
    _ax_base(ax, "Policy Entropy", ylabel="Entropy (nats)")

    ax = fig.add_subplot(gs[3, 3])
    su = np.array(history["ep_success"], dtype=float)
    roll = np.convolve(su, np.ones(min(20, n)) / min(20, n), mode="same")[:n]
    ax.bar(x[wall],  su[wall],  color=C_WALL,   alpha=0.5, width=1.0)
    ax.bar(x[~wall], su[~wall], color=C_NOWALL, alpha=0.5, width=1.0)
    ax.plot(x, roll, color=C_SUCCESS, lw=1.6, label="rolling-20")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.6)
    _ax_base(ax, "Success Rate", ylabel="Success")

    # ── Row 4: Losses + mode timeline heatmap ────────────────────────────────

    ax = fig.add_subplot(gs[4, 0])
    pl = np.array(history["ep_policy_loss"], dtype=float)
    _raw_and_smooth(ax, x, pl, C_LOSS)
    _ax_base(ax, "Policy Loss", ylabel="Loss")

    ax = fig.add_subplot(gs[4, 1])
    vl = np.array(history["ep_value_loss"], dtype=float)
    _raw_and_smooth(ax, x, vl, C_SHAPED)
    _ax_base(ax, "Value Loss", ylabel="Loss")

    ax = fig.add_subplot(gs[4, 2:4])
    dom_mode = np.argmax(mf, axis=1)

    def hex2rgb(h):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]

    rgb_img = np.array([[hex2rgb(MODE_COLORS[m]) for m in dom_mode]])
    ax.imshow(rgb_img, aspect="auto", interpolation="nearest",
              extent=[0, n, -0.5, 0.5])
    for sw in switches:
        ax.axvline(sw, color=WHITE, lw=0.8, ls=":", alpha=0.6)
    patches = [mpatches.Patch(color=MODE_COLORS[m], label=MODE_NAMES[m])
               for m in range(N_MODES)]
    ax.legend(handles=patches, fontsize=7, facecolor=PANEL,
              labelcolor=WHITE, framealpha=0.7, loc="upper right", ncol=4)
    ax.set_yticks([])
    ax.set_xlabel("Episode", color=DIM, fontsize=7.5)
    ax.tick_params(colors=DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_C)
    ax.set_facecolor(PANEL)
    ax.set_title("Dominant Mode per Episode (colour timeline)",
                 color=WHITE, fontsize=9, pad=5, fontweight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved → {out_path}")

    # --- Save individual plots ---
    base = out_path.replace(".png", "")
    individual_data = [
        ("shaped_return",  x, ret, C_RETURN, "Shaped Episode Return",  "Return"),
        ("raw_return",     x, raw, C_SHAPED, "Raw Environment Return", "Return"),
        ("stuck_fraction", x, sk,  C_STUCK,  "Stuck Fraction",         "Fraction"),
        ("episode_length", x, el,  C_RETURN, "Episode Length",         "Steps"),
        ("entropy",        x, ent, C_EPS,    "Policy Entropy",         "Nats"),
        ("policy_loss",    x, pl,  C_LOSS,   "Policy Loss",            "Loss"),
        ("value_loss",     x, vl,  C_SHAPED, "Value Loss",             "Loss"),
    ]
    for name, xi, yi, color, title, ylabel in individual_data:
        fig_i, ax_i = plt.subplots(figsize=(7, 4))
        fig_i.patch.set_facecolor(BG)
        _raw_and_smooth(ax_i, xi, yi, color)
        _ax_base(ax_i, title, ylabel=ylabel)
        plt.tight_layout()
        p = f"{base}_ppo_{name}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig_i.get_facecolor())
        plt.close()
        print(f"[plot] saved → {p}")


# =========================================================
# PPO AGENT
# =========================================================

class PPO:
    def __init__(self, env_fn, args):
        self.env_fn     = env_fn
        self.args       = args
        self.shaper     = RewardShaper()
        self.escape_fsm = StuckEscapeFSM()

        self.pnet = PolicyNetwork(
            temperature=args.temperature,
            prob_floor=args.prob_floor,
            fw_floor=args.fw_floor,
        )
        self.vnet = ValueNetwork()

        self.opt_p = optim.Adam(self.pnet.parameters(), lr=args.lr_policy)
        self.opt_v = optim.Adam(self.vnet.parameters(), lr=args.lr_value)

        self.total_steps = 0

        self.history = dict(
            ep_returns        = [],
            ep_returns_raw    = [],
            ep_lengths        = [],
            ep_success        = [],
            ep_wall           = [],
            ep_stuck_frac     = [],
            ep_fw_frac        = [],
            ep_mode_fracs     = [],
            ep_action_fracs   = [],
            ep_entropy        = [],
            ep_policy_loss    = [],
            ep_value_loss     = [],
            ep_fsm_frac       = [],
        )

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
        self.escape_fsm.reset()

        states, actions, rewards, logps, values = [], [], [], [], []
        h_p = h_v = None
        episode_success = False

        raw_returns      = 0.0
        mode_counts      = np.zeros(N_MODES, dtype=int)
        action_counts    = np.zeros(N_ACTIONS, dtype=int)
        stuck_count      = 0
        fsm_active_count = 0
        step_count       = 0

        while not done:
            s_t = torch.tensor(s, dtype=torch.float32).view(1, 1, -1)

            with torch.no_grad():
                a_net, logp, _, _, h_p, _ = self.pnet(s_t, h_p)
                v, h_v                    = self.vnet(s_t, h_v)

            h_p = (h_p[0].detach(), h_p[1].detach())
            h_v = (h_v[0].detach(), h_v[1].detach())

            stuck_now  = bool(raw_s[17] > 0.5)
            mode       = get_mode(raw_s)
            eps        = self._get_eps()
            fsm_action = self.escape_fsm.step(stuck_now)

            if fsm_action is not None:
                a              = fsm_action
                fsm_was_active = True
            elif mode == MODE_SEARCH:
                if np.random.rand() < self.args.fw_rollout_rate:
                    a = FW_IDX
                elif np.random.rand() < eps:
                    a = np.random.randint(N_ACTIONS)
                else:
                    a = int(a_net.item())
                fsm_was_active = False
            elif mode == MODE_ALIGN:
                a = np.random.randint(N_ACTIONS) if np.random.rand() < eps \
                    else int(a_net.item())
                fsm_was_active = False
            else:
                a = np.random.randint(N_ACTIONS) if np.random.rand() < eps * 0.5 \
                    else int(a_net.item())
                fsm_was_active = False

            self.total_steps += 1
            step_count       += 1
            mode_counts[mode] += 1
            action_counts[a]  += 1
            if stuck_now:      stuck_count      += 1
            if fsm_was_active: fsm_active_count += 1

            raw_s2, env_reward, done = env.step(ACTIONS[a], render=render)
            raw_s2 = np.asarray(raw_s2, dtype=np.float32)

            truncated = done and (env_reward < 1000)
            if env_reward >= 1000:
                episode_success = True

            raw_returns += float(env_reward)

            shaped_r = self.shaper.step(
                obs_prev=raw_s, obs_next=raw_s2,
                action=a, env_reward=float(env_reward),
                done=done, truncated=truncated,
                fsm_was_active=fsm_was_active,
            )

            s2 = normalise_obs(augment_obs(raw_s2))

            states.append(s)
            actions.append(a)
            rewards.append(shaped_r)
            logps.append(float(logp.item()))
            values.append(float(v.item()))

            raw_s = raw_s2
            s     = s2

        T = max(step_count, 1)
        ep_stats = dict(
            raw_return   = raw_returns,
            step_count   = step_count,
            success      = int(episode_success),
            stuck_frac   = stuck_count   / T,
            fw_frac      = action_counts[FW_IDX] / T,
            fsm_frac     = fsm_active_count / T,
            mode_fracs   = (mode_counts / T).tolist(),
            action_fracs = (action_counts / T).tolist(),
        )

        return states, actions, rewards, logps, values, episode_success, ep_stats

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

            states, actions, rewards, logps, values, success, ep_stats = \
                self.rollout(env, render=render_ep)

            adv, returns = self.compute_gae(rewards, values)

            states_t  = torch.from_numpy(np.array(states, dtype=np.float32)).unsqueeze(1)
            actions_t = torch.tensor(actions, dtype=torch.long)
            old_logps = torch.tensor(logps,   dtype=torch.float32)
            adv_t     = torch.tensor(adv,     dtype=torch.float32)
            returns_t = torch.tensor(returns, dtype=torch.float32)

            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            adv_t = torch.clamp(adv_t, -3.0, 5.0)

            entropy_coef = max(0.01, 0.05 * (1.0 - ep / self.args.total_episodes))

            ep_ploss = []
            ep_vloss = []
            ep_ent   = []

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

                ep_ploss.append(loss_p.item())
                ep_vloss.append(loss_v.item())
                ep_ent.append(ent.mean().item())

            h = self.history
            h["ep_returns"].append(sum(rewards))
            h["ep_returns_raw"].append(ep_stats["raw_return"])
            h["ep_lengths"].append(ep_stats["step_count"])
            h["ep_success"].append(ep_stats["success"])
            h["ep_wall"].append(int(use_walls))
            h["ep_stuck_frac"].append(ep_stats["stuck_frac"])
            h["ep_fw_frac"].append(ep_stats["fw_frac"])
            h["ep_fsm_frac"].append(ep_stats["fsm_frac"])
            h["ep_mode_fracs"].append(ep_stats["mode_fracs"])
            h["ep_action_fracs"].append(ep_stats["action_fracs"])
            h["ep_entropy"].append(float(np.mean(ep_ent)) if ep_ent else 0.0)
            h["ep_policy_loss"].append(float(np.mean(ep_ploss)) if ep_ploss else 0.0)
            h["ep_value_loss"].append(float(np.mean(ep_vloss)) if ep_vloss else 0.0)

            with torch.no_grad():
                _, _, _, _, _, probs_dbg = self.pnet(
                    states_t[:8], actions=actions_t[:8].unsqueeze(1))
                mean_p = probs_dbg.squeeze(1).mean(0)

            phase   = "walls" if use_walls else "     "
            act_str = "  ".join(
                f"{ACTIONS[i]}:{mean_p[i]*100:.1f}%" for i in range(N_ACTIONS))
            mf_str  = "  ".join(
                f"{MODE_NAMES[m][0]}:{ep_stats['mode_fracs'][m]*100:.0f}%"
                for m in range(N_MODES))
            flag    = "  SUCCESS" if success else ""
            fw_warn = "  ◀ FW>50%!" if mean_p[FW_IDX] > 0.50 else ""

            print(f"Ep {ep:4d} | {phase} | ret={sum(rewards):+7.2f} | "
                  f"raw={ep_stats['raw_return']:+7.1f} | "
                  f"stuck={ep_stats['stuck_frac']*100:.0f}% | "
                  f"eps={self._get_eps():.3f} | "
                  f"modes:[{mf_str}] | net:[{act_str}]{flag}{fw_warn}")

    def save(self, path):
        torch.save(self.pnet.state_dict(), path)

    def save_plots(self, path="training_plots.png"):
        plot_training(self.history, out_path=path)


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--obelix_py",       type=str,   required=True)
    parser.add_argument("--out",             type=str,   default="weights_ppo.pth")
    parser.add_argument("--plot_out",        type=str,   default="training_plots.png")
    parser.add_argument("--total_episodes",  type=int,   default=500)
    parser.add_argument("--eval_every",      type=int,   default=20)
    parser.add_argument("--seed_every",      type=int,   default=5)
    parser.add_argument("--ppo_epochs",      type=int,   default=8)

    parser.add_argument("--temperature",     type=float, default=1.5)
    parser.add_argument("--prob_floor",      type=float, default=0.03)
    parser.add_argument("--fw_floor",        type=float, default=0.0)
    parser.add_argument("--fw_rollout_rate", type=float, default=0.02)
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

    agent = PPO(env_fn, args)
    agent.train()
    agent.save(args.out)
    agent.save_plots(args.plot_out)
    print(f"[done] weights → {args.out}  |  plots → {args.plot_out}")

if __name__ == "__main__":
    main()