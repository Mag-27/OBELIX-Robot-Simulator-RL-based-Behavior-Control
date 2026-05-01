"""
Submission template for trained LSTM PPO agent.

Place your saved model file (weights_ppo.pth) in the same folder as this file.
The evaluator will import this file and call `policy(obs, rng)`.

NOTE: This agent is stateful — the LSTM hidden state persists across steps
within an episode. The hidden state is reset automatically when a new episode
begins (detected via a call to reset, or on first call).
"""

from __future__ import annotations
import os
import numpy as np

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_OBS     = 18
N_OBS_AUG = 22
N_ACTIONS = 5
N_MODES   = 4
FW_IDX    = 2

MODE_SEARCH  = 0
MODE_ALIGN   = 1
MODE_PUSH    = 2
MODE_UNSTUCK = 3

# ── Heuristic mode helpers (must match training exactly) ──────────────────────

def get_mode(obs: np.ndarray) -> int:
    stuck_flag = obs[17]
    ir         = obs[16]
    sonar_bits = obs[0:16]
    if stuck_flag:
        return MODE_UNSTUCK
    if ir:
        return MODE_PUSH
    if np.any(sonar_bits):
        return MODE_ALIGN
    return MODE_SEARCH


def mode_to_onehot(mode: int) -> np.ndarray:
    onehot = np.zeros(N_MODES, dtype=np.float32)
    onehot[mode] = 1.0
    return onehot


def augment_obs(raw_obs: np.ndarray) -> np.ndarray:
    """Append 4-dim mode one-hot → 22-dim observation."""
    mode   = get_mode(raw_obs)
    onehot = mode_to_onehot(mode)
    return np.concatenate([raw_obs, onehot])


def normalise_obs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)


# ── Model definition (must match PolicyNetwork in training code) ──────────────

_MODEL  = None   # PolicyNetwork instance
_HIDDEN = None   # (h, c) LSTM hidden state — persists within an episode
_PREV_OBS = None # used to detect episode resets


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    class PolicyNetwork(nn.Module):
        def __init__(self, stateDim=N_OBS_AUG, n_actions=N_ACTIONS,
                     hidden=128, temperature=2.0, prob_floor=0.05, fw_floor=0.10):
            super().__init__()
            self.temperature = temperature
            self.prob_floor  = prob_floor
            self.fw_floor    = fw_floor
            self.n_actions   = n_actions

            self.fc   = nn.Linear(stateDim, hidden)
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            self.out  = nn.Linear(hidden, n_actions)

            self.out.bias.data = torch.tensor([0.0, 0.0, 1.5, 0.0, 0.0])

        def forward(self, x, hidden=None):
            import torch
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

            x = torch.relu(self.fc(x))
            x, hidden = self.lstm(x, hidden)

            logits = self.out(x) / self.temperature

            probs = torch.softmax(logits, dim=-1)
            probs = (probs + self.prob_floor) / (1.0 + self.prob_floor * self.n_actions)

            # FW-specific floor
            fw_floor_t = torch.full_like(probs[..., FW_IDX], self.fw_floor)
            fw_col     = torch.where(probs[..., FW_IDX] < self.fw_floor,
                                     fw_floor_t, probs[..., FW_IDX])
            mask = torch.zeros_like(probs)
            mask[..., FW_IDX] = 1.0
            probs = probs * (1.0 - mask) + fw_col.unsqueeze(-1) * mask
            probs = probs / probs.sum(dim=-1, keepdim=True)

            greedy = probs.argmax(-1)
            return greedy, hidden, probs

    submission_dir = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(submission_dir, "weights_ppo.pth")

    import torch
    model = PolicyNetwork()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model


def reset():
    """Call this at the start of each episode to clear the LSTM state."""
    global _HIDDEN, _PREV_OBS
    _HIDDEN   = None
    _PREV_OBS = None


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Choose the best action given the current 18-dim observation.

    The LSTM hidden state is carried across calls within the same episode.
    If the observation looks like a fresh episode start, the hidden state
    is reset automatically.
    """
    global _HIDDEN, _PREV_OBS

    _load_once()

    import torch

    raw_obs = np.asarray(obs, dtype=np.float32)

    # ── Auto-detect episode reset ─────────────────────────────────────────────
    # If the evaluator doesn't call reset(), we fall back to detecting a
    # large observation jump as a heuristic episode boundary.
    if _PREV_OBS is not None:
        obs_delta = np.abs(raw_obs - _PREV_OBS).max()
        if obs_delta > 5.0:          # heuristic threshold — tune if needed
            _HIDDEN = None
    _PREV_OBS = raw_obs.copy()

    # ── Augment + normalise ───────────────────────────────────────────────────
    aug = normalise_obs(augment_obs(raw_obs))
    x   = torch.tensor(aug, dtype=torch.float32).view(1, 1, -1)  # (1, 1, 22)

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        greedy, _HIDDEN, _ = _MODEL(x, _HIDDEN)
        _HIDDEN = (_HIDDEN[0].detach(), _HIDDEN[1].detach())

    action_idx = int(greedy.item())
    return ACTIONS[action_idx]