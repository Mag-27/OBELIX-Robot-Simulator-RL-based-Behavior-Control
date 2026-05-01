"""Submission template.

Edit `policy()` to generate actions from an observation.
The evaluator will import this file and call `policy(obs, rng)`.

Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values are 0/1.
"""

from typing import Sequence

import numpy as np

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return one action for the current observation."""
    # Baseline: biased random walk that mostly goes forward.
    # Replace with your own logic.
    probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05], dtype=float)
    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]
