import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_HIDDEN = None  # LSTM hidden state

# =============================
# MODE + OBS PROCESSING
# =============================

MODE_SEARCH  = 0
MODE_ALIGN   = 1
MODE_PUSH    = 2
MODE_UNSTUCK = 3


def get_mode(obs: np.ndarray) -> int:
    if obs[17]: return MODE_UNSTUCK
    if obs[16]: return MODE_PUSH
    if np.any(obs[0:16]): return MODE_ALIGN
    return MODE_SEARCH


def mode_to_onehot(mode: int) -> np.ndarray:
    v = np.zeros(4, dtype=np.float32)
    v[mode] = 1.0
    return v


def augment_obs(raw_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([raw_obs, mode_to_onehot(get_mode(raw_obs))])


def normalise_obs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)


# =============================
# LOAD MODEL
# =============================

def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights_ppo.pth")

    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = 1.5
            self.prob_floor  = 0.03
            self.fw_floor    = 0.15

            self.fc   = nn.Linear(22, 128)
            self.lstm = nn.LSTM(128, 128, batch_first=True)
            self.out  = nn.Linear(128, 5)

            self.out.bias.data = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0])

        def forward(self, x, hidden=None):
            x = torch.relu(self.fc(x))
            x, hidden = self.lstm(x, hidden)

            logits = self.out(x) / self.temperature
            probs  = torch.softmax(logits, dim=-1)

            probs = (probs + self.prob_floor) / (1.0 + self.prob_floor * 5)

            # FW floor
            fw_col = probs[..., 2].clamp(min=self.fw_floor)
            mask   = torch.zeros_like(probs)
            mask[..., 2] = 1.0

            probs = probs * (1 - mask) + fw_col.unsqueeze(-1) * mask
            probs = probs / probs.sum(dim=-1, keepdim=True)

            return probs, hidden

    model = PolicyNetwork()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model


# =============================
# POLICY FUNCTION
# =============================

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HIDDEN

    _load_once()

    import torch

    # preprocess (MUST match training)
    x = augment_obs(obs)
    x = normalise_obs(x)

    x = torch.from_numpy(x.astype(np.float32)).view(1, 1, -1)

    with torch.no_grad():
        probs, _HIDDEN = _MODEL(x, _HIDDEN)

        # detach hidden state (important)
        _HIDDEN = (_HIDDEN[0].detach(), _HIDDEN[1].detach())

        action = torch.argmax(probs.squeeze(0).squeeze(0)).item()

    return ACTIONS[action]