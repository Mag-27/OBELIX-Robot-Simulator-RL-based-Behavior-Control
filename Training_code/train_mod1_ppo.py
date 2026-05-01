"""PPO trainer for OBELIX — matches lecture pseudocode exactly, extended with:

  • Heuristic mode augmentation  (18-dim → 22-dim observation)
  • LSTM policy & value networks
  • Mode-conditioned soft logit bias (non-trainable buffer)
  • Annealed exploration  (fw_rate, ε, search_bonus, unstuck_penalty)
  • Rotation penalty      (−0.01 per rotation action)
  • Stuck-escape FSM      (deterministic wall-recovery override)
  • Reward shaping        (mode-aware, escape sequence, contact bonuses)
  • Full reproducibility  (single master seed, isolated RNG)

Lecture-pseudocode structure preserved:
  MultiEnv        — multiprocessing worker pool (pipes + mp.Process)
  EpisodeBuffer   — fill, returnElements, updateBufferElement, reset
  PolicyNetwork   — LSTM: inputLayer → LSTM → out
  ValueNetwork    — LSTM: inputLayer → LSTM → out
  PPO             — initBookKeeping, runPPO, trainAgent, trainNetworks,
                    evaluateAgent, performBookKeeping

Run:
  python train_ppo.py --obelix_py ./obelix.py --out weights_ppo.pth
"""

from __future__ import annotations

import argparse
import random
import time
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
FW_IDX     = ACTIONS.index("FW")          # 2
ROT_IDX    = [0, 1, 3, 4]                 # all rotation action indices
N_OBS_RAW  = 18                           # raw env observation
N_OBS      = 22                           # 18 raw + 4 mode one-hot
N_ACTIONS  = 5
N_MODES    = 4

MODE_SEARCH  = 0
MODE_ALIGN   = 1
MODE_PUSH    = 2
MODE_UNSTUCK = 3

MAX_POLICY_GRAD = 0.5
MAX_VALUE_GRAD  = 0.5

SONAR_POINT_ANGLES = np.deg2rad([-90, -90, 0, 0, 0, 0, 90, 90])


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Heuristic mode helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_mode(obs: np.ndarray) -> int:
    if obs[17]:
        return MODE_UNSTUCK
    if obs[16]:
        return MODE_PUSH
    if np.any(obs[0:16]):
        return MODE_ALIGN
    return MODE_SEARCH


def mode_to_onehot(mode: int) -> np.ndarray:
    v       = np.zeros(N_MODES, dtype=np.float32)
    v[mode] = 1.0
    return v


def augment_obs(raw_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([raw_obs, mode_to_onehot(get_mode(raw_obs))])


def normalise_obs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)


def prep_obs(raw_obs: np.ndarray) -> np.ndarray:
    return normalise_obs(augment_obs(raw_obs))


# ─────────────────────────────────────────────────────────────────────────────
#  Annealed exploration schedule
# ─────────────────────────────────────────────────────────────────────────────
def linear_decay(initial: float, final: float,
                 current: int, total: int) -> float:
    t = min(current / max(total - 1, 1), 1.0)
    return initial * (1.0 - t) + final * t


# ─────────────────────────────────────────────────────────────────────────────
#  Reward Shaper
#
#  Stateful per-worker reward shaping. One instance per worker, reset each
#  episode. Called in fill() after each env.step() with the raw next obs
#  and the action that was taken.
# ─────────────────────────────────────────────────────────────────────────────
class RewardShaper:
    # SEARCH
    R_SCAN_TURN    =  0.10   # turning to scan when nothing visible
    R_BLIND_FW     = -0.30   # forward when no sonar active and not in contact

    # ALIGN
    R_CENTRING     =  0.10   # proportional to centroid alignment quality
    R_FW_ALIGNED   =  0.08   # FW when centroid is roughly forward (|err| < 45°)
    R_FW_MISALIGN  = -0.04   # FW when centroid is off to the side

    # PUSH / contact
    R_CONTACT_BONUS =  0.50  # one-shot first confirmed IR contact
    R_SUSTAIN       =  0.06  # per step while IR active and not stuck
    R_PUSH_STREAK   =  0.02  # per step, scaled by streak (cap 20)
    R_LOST_CONTACT  = -0.15  # IR drops after first touch

    # Stuck / escape sequence
    R_STUCK_BASE   = -0.20   # per stuck step
    R_STUCK_ESCAL  = -0.08   # extra per step beyond STUCK_GRACE
    STUCK_GRACE    =  2      # steps before escalation kicks in
    R_ESCAPE_TURN  =  0.15   # L45/R45 while stuck (building toward escape)
    R_ESCAPE_FW    =  0.40   # FW immediately after ≥2 L45/R45 turns while stuck

    # Spin penalty in ALIGN (should be moving forward, not spinning)
    R_SPIN_ALIGN     = -0.05
    SPIN_GRACE_ALIGN =  3

    def reset(self):
        self._had_contact        = False
        self._first_contact      = False
        self._first_detect       = False
        self._prev_ir            = 0.0
        self._push_streak        = 0
        self._consec_stuck       = 0
        self._consec_turns_align = 0
        self._escape_turns       = 0      # L45/R45 steps accumulated while stuck
        self._stuck_turned       = False  # True once ≥2 L45/R45 done while stuck

    def shape(self, raw_next: np.ndarray, action: int, env_r: float) -> float:
        """
        Returns additional shaped reward to add on top of env_r.

        raw_next : raw 18-dim observation AFTER the step
        action   : integer action index that was taken
        env_r    : original reward returned by env.step()
        """
        ir    = float(raw_next[16] > 0.5)
        stuck = float(raw_next[17] > 0.5)
        mode  = get_mode(raw_next)
        is_fw   = (action == FW_IDX)
        is_turn = (action in ROT_IDX)
        is_l45r45 = (action in {0, 4})   # only big turns count for escape

        r = 0.0

        # ── SEARCH ───────────────────────────────────────────────────────────
        if mode == MODE_SEARCH:
            if is_turn:
                r += self.R_SCAN_TURN
            if is_fw:
                r += self.R_BLIND_FW
            if not self._first_detect and np.any(raw_next[0:16] > 0.5):
                self._first_detect = True
                r += 0.30   # first sonar contact bonus

        # ── ALIGN ────────────────────────────────────────────────────────────
        if mode == MODE_ALIGN:
            unit_active = np.array([
                float(raw_next[2*i] > 0.5 or raw_next[2*i+1] > 0.5)
                for i in range(8)
            ])
            if unit_active.sum() > 0:
                cx = float(np.dot(unit_active, np.cos(SONAR_POINT_ANGLES)))
                cy = float(np.dot(unit_active, np.sin(SONAR_POINT_ANGLES)))
                theta_err = float(np.arctan2(cy, cx))
                alignment = 1.0 - abs(theta_err) / np.pi
                r += alignment * self.R_CENTRING
                if is_fw:
                    r += self.R_FW_ALIGNED if abs(theta_err) < np.pi / 4 \
                         else self.R_FW_MISALIGN
                if is_turn:
                    self._consec_turns_align += 1
                    excess = max(self._consec_turns_align - self.SPIN_GRACE_ALIGN, 0)
                    if excess > 0:
                        r += self.R_SPIN_ALIGN * excess
                else:
                    self._consec_turns_align = 0
            else:
                self._consec_turns_align = 0

        # ── First IR contact bonus ────────────────────────────────────────────
        if ir and not self._had_contact and not stuck:
            self._had_contact = True
        if self._had_contact and not self._first_contact:
            r += self.R_CONTACT_BONUS
            self._first_contact = True

        # ── PUSH ─────────────────────────────────────────────────────────────
        if mode == MODE_PUSH:
            if ir and not stuck:
                self._push_streak += 1
                r += self.R_SUSTAIN
                r += self.R_PUSH_STREAK * min(self._push_streak, 20)
            else:
                self._push_streak = max(0, self._push_streak - 1)

        # Lost contact
        if self._had_contact and self._prev_ir > 0.5 and ir < 0.5:
            r += self.R_LOST_CONTACT
        self._prev_ir = ir

        # ── Escape sequence: stuck → L45/R45 × 2 → FW ───────────────────────
        if stuck:
            if is_l45r45:
                self._escape_turns += 1
                r += self.R_ESCAPE_TURN
                if self._escape_turns >= 2:
                    self._stuck_turned = True
            elif is_fw and self._stuck_turned:
                r += self.R_ESCAPE_FW
                self._escape_turns = 0
                self._stuck_turned = False
            else:
                # L22/R22 or premature FW — reset sequence
                self._escape_turns = 0
                self._stuck_turned = False
        else:
            self._escape_turns = 0
            self._stuck_turned = False

        # ── Stuck escalation (all modes) ─────────────────────────────────────
        if stuck:
            self._consec_stuck += 1
            r += self.R_STUCK_BASE
            excess = max(self._consec_stuck - self.STUCK_GRACE, 0)
            if excess > 0:
                r += self.R_STUCK_ESCAL * excess
        else:
            self._consec_stuck = 0

        return r


# ─────────────────────────────────────────────────────────────────────────────
#  Stuck-escape FSM  (kept for eval only — no longer used during training)
# ─────────────────────────────────────────────────────────────────────────────
class StuckEscapeFSM:
    N_TURN    = 4
    COOLDOWN  = 3
    MAX_TRIES = 6

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self._reset()

    def _reset(self):
        self.state       = "IDLE"
        self.turn_count  = 0
        self.tries       = 0
        self.cooldown_ct = 0
        self.turn_dir    = 0

    def step(self, obs: np.ndarray, policy_action: int) -> int:
        stuck = bool(obs[17])

        if self.state == "COOLDOWN":
            self.cooldown_ct -= 1
            if self.cooldown_ct <= 0:
                self.state = "IDLE"
            return policy_action

        if self.state == "IDLE":
            if stuck:
                self.state      = "TURNING"
                self.turn_count = 0
                self.turn_dir   = 0 if (self.tries % 2 == 0) else 4
                self.tries     += 1
            else:
                return policy_action

        if self.state == "TURNING":
            self.turn_count += 1
            if self.turn_count >= self.N_TURN:
                self.state = "PROBE"
            return self.turn_dir

        if self.state == "PROBE":
            self.state = "PROBE_WAIT"
            return FW_IDX

        if self.state == "PROBE_WAIT":
            if not stuck:
                self.state       = "COOLDOWN"
                self.cooldown_ct = self.COOLDOWN
                self.tries       = 0
            elif self.tries >= self.MAX_TRIES:
                self._reset()
            else:
                self.state      = "TURNING"
                self.turn_count = 0
                self.turn_dir   = 4 if (self.turn_dir == 0) else 0
                self.tries     += 1
            return policy_action

        return policy_action


# ─────────────────────────────────────────────────────────────────────────────
#  Soft mode-conditioned logit bias table
# ─────────────────────────────────────────────────────────────────────────────
_MODE_LOGIT_BIAS = torch.tensor(
    [
        [ 0.15,  0.15, -0.10,  0.15,  0.15],   # SEARCH  — slight turn preference
        [ 0.20,  0.20, -0.15,  0.20,  0.20],   # ALIGN   — prefer turns to centre box
        [-0.15, -0.15,  0.40, -0.15, -0.15],   # PUSH    — prefer FW
        [ 0.35,  0.35, -0.40,  0.35,  0.35],   # UNSTUCK — prefer turns, avoid FW
    ],
    dtype=torch.float32,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Networks
# ─────────────────────────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    def __init__(
        self,
        stateDim:    int   = N_OBS,
        n_actions:   int   = N_ACTIONS,
        hDims:       tuple = (128,),
        temperature: float = 2.0,
        prob_floor:  float = 0.05,
        fw_floor:    float = 0.10,
    ):
        super().__init__()
        self.temperature = temperature
        self.prob_floor  = prob_floor
        self.fw_floor    = fw_floor
        self.n_actions   = n_actions
        self.hidden_size = hDims[0]

        self.inputLayer = nn.Linear(stateDim, hDims[0])
        self.lstm       = nn.LSTM(hDims[0], hDims[0], batch_first=True)
        self.out        = nn.Linear(hDims[0], n_actions)

        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[FW_IDX] = 1.0

        self.register_buffer("mode_logit_bias", _MODE_LOGIT_BIAS.clone())

    def forward(self, states, actions=None, hidden=None):
        ss = states if isinstance(states, torch.Tensor) \
             else torch.tensor(states, dtype=torch.float32)

        squeezed = ss.dim() == 2
        if squeezed:
            ss = ss.unsqueeze(1)

        feat        = torch.relu(self.inputLayer(ss))
        feat, hidden = self.lstm(feat, hidden)

        logits = self.out(feat)

        mode_onehot = ss[..., N_OBS_RAW:]
        mode_bias   = torch.einsum(
            "...m,mn->...n", mode_onehot, self.mode_logit_bias
        )
        logits = logits + mode_bias
        logits = logits / self.temperature

        probs = torch.softmax(logits, dim=-1)
        probs = (probs + self.prob_floor) / (1.0 + self.prob_floor * self.n_actions)

        fw_floor_t = torch.full_like(probs[..., FW_IDX], self.fw_floor)
        fw_col     = torch.where(
            probs[..., FW_IDX] < self.fw_floor, fw_floor_t, probs[..., FW_IDX]
        )
        mask              = torch.zeros_like(probs)
        mask[..., FW_IDX] = 1.0
        probs = probs * (1.0 - mask) + fw_col.unsqueeze(-1) * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist   = Categorical(probs=probs)
        greedy = probs.argmax(dim=-1)

        if actions is None:
            a = dist.sample()
        else:
            a = actions if isinstance(actions, torch.Tensor) \
                else torch.tensor(actions, dtype=torch.long)
            if squeezed and a.dim() == 1:
                a = a.unsqueeze(1)

        log_prob = dist.log_prob(a)
        entropy  = dist.entropy()

        if squeezed:
            a        = a.squeeze(1)
            log_prob = log_prob.squeeze(1)
            entropy  = entropy.squeeze(1)
            probs    = probs.squeeze(1)
            greedy   = greedy.squeeze(1)

        return a, log_prob, entropy, greedy, hidden, probs


class ValueNetwork(nn.Module):
    def __init__(self, stateDim: int = N_OBS, hDims: tuple = (128,)):
        super().__init__()
        self.hidden_size = hDims[0]
        self.inputLayer  = nn.Linear(stateDim, hDims[0])
        self.lstm        = nn.LSTM(hDims[0], hDims[0], batch_first=True)
        self.out         = nn.Linear(hDims[0], 1)

    def forward(self, states, hidden=None):
        ss = states if isinstance(states, torch.Tensor) \
             else torch.tensor(states, dtype=torch.float32)

        squeezed = ss.dim() == 2
        if squeezed:
            ss = ss.unsqueeze(1)

        feat, hidden = self.lstm(torch.relu(self.inputLayer(ss)), hidden)
        v            = self.out(feat)

        if squeezed:
            v = v.squeeze(1)

        return v, hidden


# ─────────────────────────────────────────────────────────────────────────────
#  OBELIX helpers
# ─────────────────────────────────────────────────────────────────────────────
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def create_env(OBELIX, args, seed: int):
    return OBELIX(
        scaling_factor = args.scaling_factor,
        arena_size     = args.arena_size,
        max_steps      = args.max_steps,
        wall_obstacles = args.wall_obstacles,
        difficulty     = args.difficulty,
        box_speed      = args.box_speed,
        seed           = seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  EpisodeBuffer
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeBuffer:
    def __init__(self, gamma: float, lam: float,
                 stateDim: int, numWorkers: int,
                 maxEpisodes: int, maxEpisodeSteps: int):
        self.gamma           = gamma
        self.lam             = lam
        self.stateDim        = stateDim
        self.numWorkers      = numWorkers
        self.maxEpisodes     = maxEpisodes
        self.maxEpisodeSteps = maxEpisodeSteps

        self.discounts = np.array(
            [gamma ** i for i in range(maxEpisodeSteps + 1)], dtype=np.float32)
        self.tau = np.array(
            [(gamma * lam) ** i for i in range(maxEpisodeSteps + 1)], dtype=np.float32)

        self.reset()

    def reset(self):
        E, T, S = self.maxEpisodes, self.maxEpisodeSteps, self.stateDim
        self.bufferStates   = np.zeros((E, T, S), dtype=np.float32)
        self.bufferActions  = np.zeros((E, T),    dtype=np.int64)
        self.bufferReturns  = np.zeros((E, T),    dtype=np.float32)
        self.bufferGAEs     = np.zeros((E, T),    dtype=np.float32)
        self.bufferLogp_as  = np.zeros((E, T),    dtype=np.float32)
        self.currentEpisodeIDs = list(range(self.numWorkers))
        self.episodeSteps      = np.zeros(self.maxEpisodes, dtype=np.int64)
        self.episodeRewards    = np.zeros(self.maxEpisodes, dtype=np.float32)

    def updateBufferElement(self, bufferX, episodeDetails):
        episodeIDs, episodeT = episodeDetails
        bufferElement = []
        for i, r in enumerate(bufferX[episodeIDs]):
            bufferElement.append(r[:episodeT[i]])
        return np.concatenate(bufferElement, axis=0)

    def returnElements(self):
        episodeIDs = np.where(self.episodeSteps > 0)[0]
        episodeT   = self.episodeSteps[episodeIDs]
        epDet      = (episodeIDs, episodeT)

        states  = self.updateBufferElement(self.bufferStates,  epDet)
        actions = self.updateBufferElement(
            self.bufferActions[:, :, np.newaxis], epDet).squeeze(-1)
        returns = self.updateBufferElement(self.bufferReturns, epDet)
        gaes    = self.updateBufferElement(self.bufferGAEs,    epDet)
        logp_as = self.updateBufferElement(self.bufferLogp_as, epDet)
        return states, actions, returns, gaes, logp_as

    def fill(self, envs, pNetwork: PolicyNetwork, vNetwork: ValueNetwork,
             ep: int = 0, total_episodes: int = 1,
             rng: np.random.RandomState = None,
             args=None):
        if rng is None:
            rng = np.random.RandomState(0)

        total     = max(total_episodes - 1, 1)
        eps       = linear_decay(args.eps_start,          args.eps_end,              ep, total)
        fw_rate   = linear_decay(args.fw_rollout_rate,    args.fw_rollout_rate_end,   ep, total)
        mode_rand = linear_decay(args.search_extra_eps,   0.0,                        ep, total)
        fw_pen    = linear_decay(args.unstuck_fw_penalty, 0.0,                        ep, total)

        numWorkers = self.numWorkers
        T          = self.maxEpisodeSteps

        raw_obs_per_worker = {}
        ss = []
        for id_, env in enumerate(envs):
            raw = np.asarray(env.reset(), dtype=np.float32)
            raw_obs_per_worker[id_] = raw
            ss.append(prep_obs(raw))
        ss = np.stack(ss)

        bufferFull = False
        wRewards   = np.zeros((numWorkers, T), dtype=np.float32)
        wSteps     = np.zeros(numWorkers, dtype=np.int64)

        h_p = [None] * numWorkers
        h_v = [None] * numWorkers

        # One RewardShaper per worker, reset each episode
        shapers = [RewardShaper() for _ in range(numWorkers)]
        for s in shapers:
            s.reset()

        pNetwork.eval(); vNetwork.eval()

        while not bufferFull:
            with torch.no_grad():
                actions_net = []
                logp_net    = []
                vs_net      = []

                for id_ in range(numWorkers):
                    s_t  = torch.tensor(ss[id_], dtype=torch.float32).view(1, 1, -1)
                    a_t, lp_t, _, _, h_p[id_], _ = pNetwork(s_t, hidden=h_p[id_])
                    v_t, h_v[id_]                 = vNetwork(s_t, hidden=h_v[id_])

                    if h_p[id_] is not None:
                        h_p[id_] = (h_p[id_][0].detach(), h_p[id_][1].detach())
                    if h_v[id_] is not None:
                        h_v[id_] = (h_v[id_][0].detach(), h_v[id_][1].detach())

                    actions_net.append(int(a_t.item()))
                    logp_net.append(float(lp_t.item()))
                    vs_net.append(float(v_t.squeeze().item()))

            # ── Action selection (no FSM override during training) ────────────
            actions_exec = []
            for id_ in range(numWorkers):
                raw = raw_obs_per_worker[id_]
                current_mode = get_mode(raw)

                eff_fw_rate = fw_rate
                extra_rand  = 0.0
                if current_mode == MODE_SEARCH:
                    extra_rand  = mode_rand
                elif current_mode == MODE_UNSTUCK:
                    eff_fw_rate = max(0.0, fw_rate - fw_pen)

                if rng.rand() < eff_fw_rate:
                    a = FW_IDX
                elif rng.rand() < (eps + extra_rand):
                    a = int(rng.randint(N_ACTIONS))
                else:
                    a = actions_net[id_]

                actions_exec.append(a)

            # Store transitions
            for id_ in range(numWorkers):
                epID = self.currentEpisodeIDs[id_]
                t    = wSteps[id_]
                self.bufferStates [epID, t] = ss[id_]
                self.bufferActions[epID, t] = actions_exec[id_]
                self.bufferLogp_as[epID, t] = logp_net[id_]

            sNexts = np.zeros_like(ss)
            rs     = np.zeros(numWorkers, dtype=np.float32)
            dones  = np.zeros(numWorkers, dtype=np.int32)

            for id_, env in enumerate(envs):
                obs, r, done = env.step(ACTIONS[actions_exec[id_]], render=False)
                raw_next     = np.asarray(obs, dtype=np.float32)

                # ── Reward shaping ────────────────────────────────────────────
                shaped = shapers[id_].shape(raw_next, actions_exec[id_], float(r))
                r = float(r) + shaped

                # Rotation penalty on top
                if actions_exec[id_] in ROT_IDX:
                    r -= args.rotation_penalty

                sNexts[id_]             = prep_obs(raw_next)
                rs[id_]                 = r
                dones[id_]              = int(done)
                raw_obs_per_worker[id_] = raw_next

            for id_ in range(numWorkers):
                wRewards[id_, wSteps[id_]] = rs[id_]

            for id_ in range(numWorkers):
                if wSteps[id_] + 1 == T:
                    dones[id_] = 1

            if dones.sum():
                dones_ids = np.where(dones)[0]
                nValues   = np.zeros(numWorkers, dtype=np.float32)
                with torch.no_grad():
                    for id_ in dones_ids:
                        sn_t = torch.tensor(
                            sNexts[id_], dtype=torch.float32).view(1, 1, -1)
                        v_n, _ = vNetwork(sn_t, hidden=h_v[id_])
                        nValues[id_] = float(v_n.squeeze().item())

                for id_ in dones_ids:
                    raw = np.asarray(envs[id_].reset(), dtype=np.float32)
                    raw_obs_per_worker[id_] = raw
                    sNexts[id_]             = prep_obs(raw)
                    h_p[id_] = None
                    h_v[id_] = None
                    shapers[id_].reset()   # reset shaper state for new episode

                for id_ in dones_ids:
                    epID = self.currentEpisodeIDs[id_]
                    ep_T = int(wSteps[id_]) + 1
                    self.episodeSteps[epID]   = ep_T
                    self.episodeRewards[epID] = float(wRewards[id_, :ep_T].sum())

                    epRewards   = np.append(wRewards[id_, :ep_T], nValues[id_])
                    epDiscounts = self.discounts[:ep_T + 1]
                    epReturns   = [
                        np.sum(epDiscounts[:ep_T + 1 - t] * epRewards[t:])
                        for t in range(ep_T)
                    ]
                    self.bufferReturns[epID, :ep_T] = np.array(epReturns, dtype=np.float32)

                    epStates = self.bufferStates[epID, :ep_T]
                    ep_vs    = []
                    with torch.no_grad():
                        h_tmp = None
                        for t_step in range(ep_T):
                            s_t = torch.tensor(
                                epStates[t_step], dtype=torch.float32).view(1, 1, -1)
                            v_t, h_tmp = vNetwork(s_t, hidden=h_tmp)
                            ep_vs.append(float(v_t.squeeze().item()))
                    epV      = np.array(ep_vs, dtype=np.float32)
                    epValues = np.append(epV, nValues[id_])
                    epTau    = self.tau[:ep_T]
                    deltas   = epRewards[:-1] + self.gamma * epValues[1:] - epValues[:-1]
                    gaes     = [
                        np.sum(epTau[:ep_T - t] * deltas[t:]) for t in range(ep_T)
                    ]
                    self.bufferGAEs[epID, :ep_T] = np.array(gaes, dtype=np.float32)

                    new_epID = max(self.currentEpisodeIDs) + 1
                    if new_epID >= self.maxEpisodes:
                        bufferFull = True
                        break
                    self.currentEpisodeIDs[id_] = new_epID
                    wRewards[id_] = 0.0
                    wSteps[id_]   = -1

            ss      = sNexts
            wSteps += 1

        pNetwork.train(); vNetwork.train()


# ─────────────────────────────────────────────────────────────────────────────
#  PPO class
# ─────────────────────────────────────────────────────────────────────────────
class PPO:
    def __init__(
        self,
        env,
        gamma:                float,
        lam:                  float,
        beta:                 float,
        numWorkers:           int,
        maxEpisodes:          int,
        maxEpisodeSteps:      int,
        updateFrequency:      int,
        policyOptimizerFn,
        policyOptimizerLR:    float,
        policyOptimizationEpochs: int,
        policyClipRange:      float,
        policySampleRatio:    float,
        policyStoppingKL:     float,
        valueOptimizerFn,
        valueOptimizerLR:     float,
        valueOptimizationEpochs: int,
        valueClipRange:       float,
        valueSampleRatio:     float,
        valueStoppingMSE:     float,
        MAX_EVAL_EPISODE:     int,
        stateDim:             int   = N_OBS,
        n_actions:            int   = N_ACTIONS,
        hDims_policy:         tuple = (128,),
        hDims_value:          tuple = (128,),
        MAX_TRAIN_EPISODES:   int   = 2000,
        seed:                 int   = 42,
        eval_render:          bool  = False,
        eval_env_fn                 = None,
        args                        = None,
    ):
        self.env                       = env
        self.gamma                     = gamma
        self.lam                       = lam
        self.beta                      = beta
        self.numWorkers                = numWorkers
        self.maxEpisodes               = maxEpisodes
        self.maxEpisodeSteps           = maxEpisodeSteps
        self.policyOptimizationEpochs  = policyOptimizationEpochs
        self.policyClipRange           = policyClipRange
        self.policySampleRatio         = policySampleRatio
        self.policyStoppingKL          = policyStoppingKL
        self.valueOptimizationEpochs   = valueOptimizationEpochs
        self.valueClipRange            = valueClipRange
        self.valueSampleRatio          = valueSampleRatio
        self.valueStoppingMSE          = valueStoppingMSE
        self.MAX_EVAL_EPISODE          = MAX_EVAL_EPISODE
        self.MAX_TRAIN_EPISODES        = MAX_TRAIN_EPISODES
        self.seed                      = seed
        self.eval_render               = eval_render
        self.eval_env_fn               = eval_env_fn
        self.args                      = args

        self.rng = np.random.RandomState(seed)

        self.pNetwork = PolicyNetwork(
            stateDim    = stateDim,
            n_actions   = n_actions,
            hDims       = hDims_policy,
            temperature = args.temperature,
            prob_floor  = args.prob_floor,
            fw_floor    = args.fw_floor,
        )
        self.vNetwork = ValueNetwork(stateDim=stateDim, hDims=hDims_value)

        self.policyOptimizerFn = policyOptimizerFn(
            self.pNetwork.parameters(), lr=policyOptimizerLR)
        self.valueOptimizerFn  = valueOptimizerFn(
            self.vNetwork.parameters(), lr=valueOptimizerLR)

        self.rBuffer = EpisodeBuffer(
            gamma           = gamma,
            lam             = lam,
            stateDim        = stateDim,
            numWorkers      = numWorkers,
            maxEpisodes     = maxEpisodes,
            maxEpisodeSteps = maxEpisodeSteps,
        )

        self.total_eps_so_far = 0
        self.initBookKeeping()
        self._print_config()

    def _print_config(self):
        a = self.args
        print("=" * 72)
        print("  PPO + LSTM + Heuristic Modes + Reward Shaping  |  OBELIX")
        print("=" * 72)
        for k, v in [
            ("seed",                  self.seed),
            ("obs_dim",               N_OBS),
            ("MAX_TRAIN_EPISODES",    self.MAX_TRAIN_EPISODES),
            ("maxEpisodes/fill",      self.maxEpisodes),
            ("numWorkers",            self.numWorkers),
            ("gamma",                 self.gamma),
            ("lam",                   self.lam),
            ("beta (entropy)",        self.beta),
            ("temperature",           a.temperature),
            ("prob_floor",            a.prob_floor),
            ("fw_floor",              a.fw_floor),
            ("eps_start/end",         f"{a.eps_start}/{a.eps_end}"),
            ("fw_rollout_rate s/e",   f"{a.fw_rollout_rate}/{a.fw_rollout_rate_end}"),
            ("search_extra_eps",      a.search_extra_eps),
            ("unstuck_fw_penalty",    a.unstuck_fw_penalty),
            ("rotation_penalty",      a.rotation_penalty),
            ("policyClipRange",       self.policyClipRange),
            ("valueClipRange",        self.valueClipRange),
            ("policy lr",             a.lr_policy),
            ("value  lr",             a.lr_value),
        ]:
            print(f"  {k:<28} = {v}")
        print("=" * 72, flush=True)

    def initBookKeeping(self):
        self.trainEpisodeRewards: list[float] = []
        self.evalEpisodeRewards:  list[float] = []
        self.policyLosses:        list[float] = []
        self.valueLosses:         list[float] = []
        self.entropies:           list[float] = []
        self.wallTimes:           list[float] = []
        self._t0 = time.perf_counter()

    def runPPO(self):
        resultTrain = self.trainAgent()
        evalMean, evalStd = self.evaluateAgent()
        trainingTime  = time.perf_counter() - self._t0
        wallclockTime = self.wallTimes[-1] if self.wallTimes else 0.0
        return resultTrain, evalMean, trainingTime, wallclockTime

    def _make_envs(self, use_walls: bool, offset: int = 0):
        """Create a fresh set of worker envs with the given wall setting."""
        args = self.args
        # Temporarily override wall_obstacles
        prev = args.wall_obstacles
        args.wall_obstacles = use_walls
        envs = [self.eval_env_fn(seed=self.seed + offset + i)
                for i in range(self.numWorkers)]
        args.wall_obstacles = prev
        return envs

    def _close_envs(self, envs):
        for env in envs:
            try:
                env.close()
            except Exception:
                pass

    def trainAgent(self):
        results    = []
        totalEps   = 0
        fill_count = 0
        use_walls  = False   # start without walls; toggle every 10 episodes

        envs = self._make_envs(use_walls, offset=0)

        while True:
            self.rBuffer.fill(
                envs,
                self.pNetwork,
                self.vNetwork,
                ep             = totalEps,
                total_episodes = self.MAX_TRAIN_EPISODES,
                rng            = self.rng,
                args           = self.args,
            )
            self.trainNetworks()

            valid = np.where(self.rBuffer.episodeSteps > 0)[0]
            for ep in valid:
                self.trainEpisodeRewards.append(
                    float(self.rBuffer.episodeRewards[ep]))
            totalEps            += len(valid)
            self.total_eps_so_far = totalEps
            fill_count           += 1

            self.rBuffer.reset()
            self.performBookKeeping(train=True)

            # Toggle wall_obstacles every 10 episodes
            new_use_walls = (totalEps // 10) % 2 == 1
            if new_use_walls != use_walls:
                self._close_envs(envs)
                use_walls = new_use_walls
                envs = self._make_envs(use_walls, offset=totalEps)
                phase = "walls ON" if use_walls else "walls OFF"
                print(f"  [phase] ep={totalEps} → {phase}", flush=True)

            timeOut = (time.perf_counter() - self._t0) > 3600 * 4
            reachedMaxEp = totalEps >= self.MAX_TRAIN_EPISODES
            reachedGoalReward = (
                len(self.trainEpisodeRewards) >= 10 and
                np.mean(self.trainEpisodeRewards[-10:]) > 1800
            )

            if timeOut or reachedGoalReward or reachedMaxEp:
                break

            results.append((fill_count, totalEps, float(
                np.mean(self.trainEpisodeRewards[-self.maxEpisodes:]))))

        self._close_envs(envs)
        return results

    def trainNetworks(self):
        ss, actions, rs, gaes, logps = self.rBuffer.returnElements()

        ss_t      = torch.tensor(ss,      dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        returns_t = torch.tensor(rs,      dtype=torch.float32)
        gaes_t    = torch.tensor(gaes,    dtype=torch.float32)
        old_logps = torch.tensor(logps,   dtype=torch.float32)

        with torch.no_grad():
            old_values, _ = self.vNetwork(ss_t)
            old_values     = old_values.squeeze(-1)

        gaes_t = (gaes_t - gaes_t.mean()) / (gaes_t.std() + 1e-8)
        gaes_t = torch.clamp(gaes_t, min=-2.0, max=5.0)

        if self.total_eps_so_far < 100:
            gaes_t = torch.clamp(gaes_t, min=-1.0)

        nSamples         = len(ss_t)
        batchSize_policy = max(1, int(self.policySampleRatio * nSamples))
        batchSize_value  = max(1, int(self.valueSampleRatio  * nSamples))

        entropy_coef = max(0.02,
            self.beta * (1.0 - self.total_eps_so_far / self.MAX_TRAIN_EPISODES))

        for _ in range(self.policyOptimizationEpochs):
            idx = torch.randperm(nSamples)[:batchSize_policy]

            s_b        = ss_t[idx]
            a_b        = actions_t[idx]
            adv_b      = gaes_t[idx]
            logp_old_b = old_logps[idx]

            _, logp, entropy, _, _, probs_batch = self.pNetwork(s_b, a_b)

            ratio = torch.exp(logp - logp_old_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio,
                                 1.0 - self.policyClipRange,
                                 1.0 + self.policyClipRange) * adv_b

            fw_prob       = probs_batch[..., FW_IDX].mean()
            entropy_bonus = 2.0 if fw_prob < 0.1 else 1.0

            action_freq = probs_batch.mean(dim=0)
            uniform     = torch.ones_like(action_freq) / N_ACTIONS
            div_loss    = ((action_freq - uniform) ** 2).mean()

            policy_loss  = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy_bonus * entropy_coef * entropy.mean()
            loss         = policy_loss + entropy_loss + 0.1 * div_loss

            self.policyOptimizerFn.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.pNetwork.parameters(), MAX_POLICY_GRAD)
            self.policyOptimizerFn.step()

            self.policyLosses.append(policy_loss.item())
            self.entropies.append(entropy.mean().item())

            with torch.no_grad():
                _, new_logp_all, _, _, _, _ = self.pNetwork(ss_t, actions_t)
                kl = (old_logps - new_logp_all).mean()

            if kl > self.policyStoppingKL:
                break

        for _ in range(self.valueOptimizationEpochs):
            idx = torch.randperm(nSamples)[:batchSize_value]

            s_b     = ss_t[idx]
            r_b     = returns_t[idx]
            v_old_b = old_values[idx]

            v_pred, _ = self.vNetwork(s_b)
            v_pred     = v_pred.squeeze(-1)

            v_clipped      = v_old_b + torch.clamp(
                v_pred - v_old_b, -self.valueClipRange, self.valueClipRange)
            loss_unclipped = (v_pred    - r_b) ** 2
            loss_clipped   = (v_clipped - r_b) ** 2
            value_loss     = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

            self.valueOptimizerFn.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.vNetwork.parameters(), MAX_VALUE_GRAD)
            self.valueOptimizerFn.step()

            self.valueLosses.append(value_loss.item())

            with torch.no_grad():
                v_all, _ = self.vNetwork(ss_t)
                mse = ((v_all.squeeze(-1) - returns_t) ** 2).mean()

            if mse < self.valueStoppingMSE:
                break

    def evaluateAgent(self):
        """Greedy rollouts with FSM override (eval only)."""
        rewards = []
        self.pNetwork.eval()
        eval_fsm   = StuckEscapeFSM(np.random.RandomState(self.seed + 99999))
        eval_shaper = RewardShaper()

        with torch.no_grad():
            for e in range(self.MAX_EVAL_EPISODE):
                rs   = 0.0
                env  = self.eval_env_fn(seed=self.seed + 90_000 + e)
                raw  = np.asarray(env.reset(), dtype=np.float32)
                s    = prep_obs(raw)
                done = False
                h_p  = None
                eval_fsm._reset()
                eval_shaper.reset()

                for c in count():
                    s_t = torch.tensor(s, dtype=torch.float32).view(1, 1, -1)
                    _, _, _, greedy, h_p, _ = self.pNetwork(s_t, hidden=h_p)
                    if h_p is not None:
                        h_p = (h_p[0].detach(), h_p[1].detach())

                    a = eval_fsm.step(raw, int(greedy.item()))

                    raw2, r, done = env.step(ACTIONS[a], render=self.eval_render)
                    raw2 = np.asarray(raw2, dtype=np.float32)
                    r    = float(r) + eval_shaper.shape(raw2, a, float(r))
                    raw  = raw2
                    s    = prep_obs(raw)
                    rs  += r

                    if done or c >= self.maxEpisodeSteps - 1:
                        rewards.append(rs)
                        break

        self.evalEpisodeRewards.extend(rewards)
        self.performBookKeeping(train=False)
        self.pNetwork.train()
        return float(np.mean(rewards)), float(np.std(rewards))

    def performBookKeeping(self, train: bool = True):
        now = time.perf_counter() - self._t0
        self.wallTimes.append(now)

        if train and self.trainEpisodeRewards:
            recent = self.trainEpisodeRewards[-self.maxEpisodes:]
            r_arr  = np.array(recent)

            ep  = self.total_eps_so_far
            tot = self.MAX_TRAIN_EPISODES
            eps_now = linear_decay(self.args.eps_start,       self.args.eps_end,            ep, tot)
            fw_now  = linear_decay(self.args.fw_rollout_rate, self.args.fw_rollout_rate_end, ep, tot)
            mr_now  = linear_decay(self.args.search_extra_eps, 0.0,                         ep, tot)

            SEP = "─" * 72
            print(f"\n{SEP}")
            print(f"  ▶  Total eps={len(self.trainEpisodeRewards)}"
                  f" / {self.MAX_TRAIN_EPISODES}"
                  f"   ({now:.0f}s elapsed)")
            print(f"  RETURNS      "
                  f"mean={r_arr.mean():+9.2f}  std={r_arr.std():7.2f}  "
                  f"min={r_arr.min():+9.2f}  max={r_arr.max():+9.2f}")
            print(f"  EXPLORATION  "
                  f"eps={eps_now:.3f}  fw_rate={fw_now:.3f}  "
                  f"mode_rand={mr_now:.3f}")
            if self.policyLosses:
                pl  = np.array(self.policyLosses[-20:])
                vl  = np.array(self.valueLosses[-20:]) if self.valueLosses else np.array([0.0])
                ent = np.array(self.entropies[-20:])
                print(f"  POLICY LOSS  mean={pl.mean():+9.4f}")
                print(f"  VALUE  LOSS  mean={vl.mean():+9.4f}")
                print(f"  ENTROPY      mean={ent.mean():.4f}  "
                      f"({'exploring' if ent.mean() > 1.0 else 'converging'})")
            print(SEP, flush=True)

        elif not train and self.evalEpisodeRewards:
            recent_eval = self.evalEpisodeRewards[-self.MAX_EVAL_EPISODE:]
            print(f"  [eval]  mean={np.mean(recent_eval):+.2f}  "
                  f"std={np.std(recent_eval):.2f}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--obelix_py",             type=str,   required=True)
    ap.add_argument("--out",                   type=str,   default="weights_ppo.pth")
    ap.add_argument("--seed",                  type=int,   default=42)
    ap.add_argument("--total_episodes",        type=int,   default=2000)
    ap.add_argument("--episodes_per_fill",     type=int,   default=16)
    ap.add_argument("--max_steps",             type=int,   default=200)
    ap.add_argument("--num_workers",           type=int,   default=4)
    ap.add_argument("--eval_episodes",         type=int,   default=5)
    ap.add_argument("--eval_render",           action="store_true")
    ap.add_argument("--difficulty",            type=int,   default=0)
    ap.add_argument("--wall_obstacles",        action="store_true")
    ap.add_argument("--box_speed",             type=int,   default=2)
    ap.add_argument("--scaling_factor",        type=int,   default=5)
    ap.add_argument("--arena_size",            type=int,   default=500)
    ap.add_argument("--gamma",                 type=float, default=0.98)
    ap.add_argument("--lam",                   type=float, default=0.95)
    ap.add_argument("--beta",                  type=float, default=0.05)
    ap.add_argument("--lr_policy",             type=float, default=1e-4)
    ap.add_argument("--lr_value",              type=float, default=1e-3)
    ap.add_argument("--policy_epochs",         type=int,   default=10)
    ap.add_argument("--value_epochs",          type=int,   default=10)
    ap.add_argument("--policy_clip",           type=float, default=0.2)
    ap.add_argument("--value_clip",            type=float, default=0.2)
    ap.add_argument("--policy_sample_ratio",   type=float, default=0.8)
    ap.add_argument("--value_sample_ratio",    type=float, default=0.8)
    ap.add_argument("--policy_stopping_kl",    type=float, default=0.03)
    ap.add_argument("--value_stopping_mse",    type=float, default=1e6)
    ap.add_argument("--hDims_policy",          type=int,   nargs="+", default=[128])
    ap.add_argument("--hDims_value",           type=int,   nargs="+", default=[128])
    ap.add_argument("--temperature",           type=float, default=2.0)
    ap.add_argument("--prob_floor",            type=float, default=0.05)
    ap.add_argument("--fw_floor",              type=float, default=0.10)
    ap.add_argument("--eps_start",             type=float, default=0.10)
    ap.add_argument("--eps_end",               type=float, default=0.01)
    ap.add_argument("--fw_rollout_rate",       type=float, default=0.40)
    ap.add_argument("--fw_rollout_rate_end",   type=float, default=0.05)
    ap.add_argument("--search_extra_eps",      type=float, default=0.10)
    ap.add_argument("--unstuck_fw_penalty",    type=float, default=0.25)
    ap.add_argument("--rotation_penalty",      type=float, default=0.01)

    args = ap.parse_args()

    set_global_seeds(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    def env_fn(seed: int):
        return create_env(OBELIX, args, seed)

    eval_env = env_fn(seed=args.seed)

    agent = PPO(
        env                      = eval_env,
        gamma                    = args.gamma,
        lam                      = args.lam,
        beta                     = args.beta,
        numWorkers               = args.num_workers,
        maxEpisodes              = args.episodes_per_fill,
        maxEpisodeSteps          = args.max_steps,
        updateFrequency          = 1,
        policyOptimizerFn        = optim.Adam,
        policyOptimizerLR        = args.lr_policy,
        policyOptimizationEpochs = args.policy_epochs,
        policyClipRange          = args.policy_clip,
        policySampleRatio        = args.policy_sample_ratio,
        policyStoppingKL         = args.policy_stopping_kl,
        valueOptimizerFn         = optim.Adam,
        valueOptimizerLR         = args.lr_value,
        valueOptimizationEpochs  = args.value_epochs,
        valueClipRange           = args.value_clip,
        valueSampleRatio         = args.value_sample_ratio,
        valueStoppingMSE         = args.value_stopping_mse,
        MAX_EVAL_EPISODE         = args.eval_episodes,
        stateDim                 = N_OBS,
        n_actions                = N_ACTIONS,
        hDims_policy             = tuple(args.hDims_policy),
        hDims_value              = tuple(args.hDims_value),
        MAX_TRAIN_EPISODES       = args.total_episodes,
        seed                     = args.seed,
        eval_render              = args.eval_render,
        eval_env_fn              = env_fn,
        args                     = args,
    )

    resultTrain, final_eval, training_time, wallclock = agent.runPPO()

    torch.save(agent.pNetwork.cpu().state_dict(), args.out)
    print(f"\n[train_ppo] Saved        → {args.out}")
    print(f"[train_ppo] Final eval   : {final_eval:+.2f}")
    print(f"[train_ppo] Training time: {training_time:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()