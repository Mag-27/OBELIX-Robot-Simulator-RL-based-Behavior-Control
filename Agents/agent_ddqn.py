"""
Submission template (DDQN - EXACT ARCHITECTURE MATCH).

Architecture mirrors createValueNetwork() from the DDQN trainer:
    Linear(18, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 5)

Weights file: ddqn_weights.pth
"""

import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS   = ("L45", "L22", "FW", "R22", "R45")
N_OBS     = 18
N_ACTIONS = 5

_MODEL = None


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "Weights/weights_ddqn.pth")

    model = nn.Sequential(
        nn.Linear(N_OBS, 128), nn.ReLU(),
        nn.Linear(128, 128),   nn.ReLU(),
        nn.Linear(128, N_ACTIONS),
    )

    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0)
        action   = int(torch.argmax(q_values).item())
    return ACTIONS[action]