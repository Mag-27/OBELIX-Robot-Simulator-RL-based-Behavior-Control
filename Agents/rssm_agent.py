"""
Submission Template - PPO + World Model (RSSM)

IMPORTANT:
- Requires TWO weight files in same directory:
    1. weights_rssm_ppo.pth  (policy)
    2. world_model.pth       (world model)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# CONSTANTS
# =========================================================
ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_OBS = 18
N_ACTIONS = 5

Z_DIM = 32
HIDDEN_DIM = 64

_POLICY = None
_WM = None
_HX = None
_PREV_ACTION = None


# =========================================================
# WORLD MODEL (MUST MATCH TRAINING)
# =========================================================
class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(N_OBS + N_ACTIONS, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.pred_head = nn.Linear(HIDDEN_DIM, N_OBS)
        self.encode_head = nn.Linear(HIDDEN_DIM, Z_DIM)

    def forward(self, obs_seq, act_seq, hx=None):
        x = torch.cat([obs_seq, act_seq], dim=-1)
        x = F.relu(self.encoder(x))
        lstm_out, hx = self.lstm(x, hx)
        pred_obs = torch.sigmoid(self.pred_head(lstm_out))
        z = self.encode_head(lstm_out)
        return pred_obs, z, hx

    def init_hx(self):
        h = torch.zeros(1, 1, HIDDEN_DIM)
        c = torch.zeros(1, 1, HIDDEN_DIM)
        return (h, c)

    def encode_step(self, obs, action_idx, hx):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        act = torch.zeros(1, 1, N_ACTIONS)
        act[0, 0, action_idx] = 1.0

        _, z, hx = self.forward(obs_t, act, hx)
        return z.squeeze(0).squeeze(0), hx


# =========================================================
# POLICY NETWORK (USES z_t)
# =========================================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputLayer = nn.Linear(Z_DIM, 64)
        self.hLayers = nn.ModuleList([
            nn.Linear(64, 64)
        ])
        self.out = nn.Linear(64, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.inputLayer(x))
        for h in self.hLayers:
            x = F.relu(h(x))
        return self.out(x)


# =========================================================
# LOAD MODELS
# =========================================================
def _load_once():
    global _POLICY, _WM, _HX, _PREV_ACTION

    if _POLICY is not None:
        return

    base = os.path.dirname(__file__)

    policy_path = os.path.join(base, "weights_rssm_ppo.pth")
    wm_path = os.path.join(base, "world_model.pth")

    if not os.path.exists(policy_path):
        raise FileNotFoundError(policy_path)

    if not os.path.exists(wm_path):
        raise FileNotFoundError(wm_path)

    # Load policy
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
    policy.eval()

    # Load world model
    wm = WorldModel()
    wm.load_state_dict(torch.load(wm_path, map_location="cpu"))
    wm.eval()

    _POLICY = policy
    _WM = wm

    # Initialize hidden state
    _HX = wm.init_hx()

    # Initial action = FW (important!)
    _PREV_ACTION = 2


# =========================================================
# POLICY FUNCTION
# =========================================================
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HX, _PREV_ACTION

    _load_once()

    if not isinstance(obs, np.ndarray):
        obs = np.array(obs, dtype=np.float32)

    obs = obs.astype(np.float32)

    # 🔥 RESET DETECTION (VERY IMPORTANT)
    # If all sensors reset or first call, reset hx
    if np.all(obs == 0):
        _HX = _WM.init_hx()
        _PREV_ACTION = 2  # FW

    # Get belief state z_t
    with torch.no_grad():
        z_t, _HX = _WM.encode_step(obs, _PREV_ACTION, _HX)
        z_t = z_t.unsqueeze(0)

        logits = _POLICY(z_t)
        action_idx = torch.argmax(logits, dim=1).item()

    _PREV_ACTION = action_idx

    return ACTIONS[action_idx]