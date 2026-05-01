"""
Submission template (NO weights required).

Use this template if your agent does not use any trained model.
Do NOT load torch or any weight files.

The action is decided directly using simple logic or randomness.

The evaluator will import this file and call `policy(obs, rng)`.
"""

from typing import Sequence
import numpy as np

# All possible actions
ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Choose an action directly from the observation.
    No neural network or weights are used.
    """

    # Example baseline: mostly move forward
    probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05], dtype=float)

    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]
