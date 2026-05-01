"""
Submission template (PPO LSTM + mode augmentation + reward shaping).
Matches training architecture EXACTLY.
"""

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

N_OBS_RAW = 18
N_OBS     = 22
N_ACTIONS = 5
N_MODES   = 4
FW_IDX    = 2

_MODEL  = None
_HIDDEN = None
_FSM    = None


# ─────────────────────────────────────────────────────────────
# Mode helpers (MUST match training exactly)
# ─────────────────────────────────────────────────────────────
def get_mode(obs):
    if obs[17]: return 3  # UNSTUCK
    if obs[16]: return 2  # PUSH
    if np.any(obs[0:16]): return 1  # ALIGN
    return 0              # SEARCH


def mode_to_onehot(mode):
    v = np.zeros(N_MODES, dtype=np.float32)
    v[mode] = 1.0
    return v


def augment_obs(raw_obs):
    return np.concatenate([raw_obs, mode_to_onehot(get_mode(raw_obs))])


def normalise_obs(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def prep_obs(raw_obs):
    return normalise_obs(augment_obs(raw_obs))


# ─────────────────────────────────────────────────────────────
# Stuck-escape FSM (matches evaluateAgent in trainer)
# ─────────────────────────────────────────────────────────────
class StuckEscapeFSM:
    N_TURN    = 4
    COOLDOWN  = 3
    MAX_TRIES = 6

    def __init__(self):
        self._reset()

    def _reset(self):
        self.state       = "IDLE"
        self.turn_count  = 0
        self.tries       = 0
        self.cooldown_ct = 0
        self.turn_dir    = 0

    def step(self, obs, policy_action):
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


# ─────────────────────────────────────────────────────────────
# Policy Network (EXACT match to trainer)
# ─────────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 2.0
        self.prob_floor  = 0.05
        self.fw_floor    = 0.10

        self.inputLayer = nn.Linear(N_OBS, 128)
        self.lstm       = nn.LSTM(128, 128, batch_first=True)
        self.out        = nn.Linear(128, N_ACTIONS)

        self.register_buffer("mode_logit_bias", torch.tensor([
            [ 0.15,  0.15, -0.10,  0.15,  0.15],
            [ 0.20,  0.20, -0.15,  0.20,  0.20],
            [-0.15, -0.15,  0.40, -0.15, -0.15],
            [ 0.35,  0.35, -0.40,  0.35,  0.35],
        ], dtype=torch.float32))

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        feat, hidden = self.lstm(torch.relu(self.inputLayer(x)), hidden)
        logits       = self.out(feat)

        mode_onehot = x[..., N_OBS_RAW:]
        mode_bias   = torch.einsum("...m,mn->...n", mode_onehot, self.mode_logit_bias)
        logits      = (logits + mode_bias) / self.temperature

        probs = torch.softmax(logits, dim=-1)
        probs = (probs + self.prob_floor) / (1.0 + self.prob_floor * N_ACTIONS)

        fw      = probs[..., FW_IDX].clamp(min=self.fw_floor)
        mask    = torch.zeros_like(probs)
        mask[..., FW_IDX] = 1.0
        probs   = probs * (1 - mask) + fw.unsqueeze(-1) * mask
        probs   = probs / probs.sum(dim=-1, keepdim=True)

        return probs, hidden


# ─────────────────────────────────────────────────────────────
# Load model once
# ─────────────────────────────────────────────────────────────
def _load_once():
    global _MODEL, _FSM
    if _MODEL is not None:
        return

    wpath = os.path.join(os.path.dirname(__file__), "weights_ppo.pth")
    model = PolicyNetwork()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model
    _FSM   = StuckEscapeFSM()


# ─────────────────────────────────────────────────────────────
# Episode reset hook — call this at the start of each episode
# ─────────────────────────────────────────────────────────────
def reset():
    global _HIDDEN, _FSM
    _HIDDEN = None
    if _FSM is not None:
        _FSM._reset()


# ─────────────────────────────────────────────────────────────
# Policy function
# ─────────────────────────────────────────────────────────────
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HIDDEN

    _load_once()

    x = torch.from_numpy(prep_obs(obs).astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        probs, _HIDDEN = _MODEL(x, _HIDDEN)
        if _HIDDEN is not None:
            _HIDDEN = (_HIDDEN[0].detach(), _HIDDEN[1].detach())
        action = torch.argmax(probs.squeeze()).item()

    # FSM override for stuck recovery
    action = _FSM.step(obs, action)

    return ACTIONS[action]