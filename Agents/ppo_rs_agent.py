"""
Submission template (PPO with RewardShaper - EXACT ARCHITECTURE MATCH).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_OBS = 18
N_ACTIONS = 5

_MODEL = None


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "Weights/weights_ppo_rs.pth")

    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputLayer = nn.Linear(N_OBS, 64)
            self.hLayers = nn.ModuleList([nn.Linear(64, 64)])
            self.out = nn.Linear(64, N_ACTIONS)

        def forward(self, x):
            x = F.relu(self.inputLayer(x))
            for h in self.hLayers:
                x = F.relu(h(x))
            logits = self.out(x)
            return logits

    model = PolicyNetwork()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits = _MODEL(x).squeeze(0)
        action = torch.argmax(logits).item()
    return ACTIONS[action]