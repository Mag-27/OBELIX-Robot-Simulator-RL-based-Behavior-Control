from __future__ import annotations
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli

ACTIONS      = ["L45", "L22", "FW", "R22", "R45"]
FW_IDX       = ACTIONS.index("FW")        # 2
ROT_IDX      = [0, 1, 3, 4]
N_OBS        = 18
N_OBS_AUG    = 22                          # 18 raw + 4 option one-hot
N_ACTIONS    = 5
N_OPTIONS    = 4                           # SEARCH / ALIGN / PUSH / UNSTUCK

OPTION_NAMES = ["SEARCH", "ALIGN", "PUSH", "UNSTUCK"]

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def option_to_onehot(option: int, n: int = N_OPTIONS) -> np.ndarray:
    v         = np.zeros(n, dtype=np.float32)
    v[option] = 1.0
    return v


def augment_obs(raw_obs: np.ndarray, option: int) -> np.ndarray:
    return np.concatenate([raw_obs, option_to_onehot(option)])


def normalise_obs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)


def prep_obs(raw_obs: np.ndarray, option: int) -> np.ndarray:
    """Augment with option one-hot then normalise (for intra-option policy)."""
    return normalise_obs(augment_obs(raw_obs, option))


def prep_raw(raw_obs: np.ndarray) -> np.ndarray:
    """Normalise without augmentation (for option-level networks)."""
    return normalise_obs(raw_obs)


# ─────────────────────────────────────────────────────────────────────────────
# SOFT INTRA-OPTION LOGIT BIAS  (warm-start, non-trainable)
# ─────────────────────────────────────────────────────────────────────────────

_OPTION_LOGIT_BIAS = torch.tensor(
    [
        #  L45    L22    FW     R22    R45
        [ 0.15,  0.15, -0.10,  0.15,  0.15],   # option 0 – SEARCH
        [ 0.20,  0.20, -0.15,  0.20,  0.20],   # option 1 – ALIGN
        [-0.15, -0.15,  0.40, -0.15, -0.15],   # option 2 – PUSH
        [ 0.35,  0.35, -0.40,  0.35,  0.35],   # option 3 – UNSTUCK
    ],
    dtype=torch.float32,
)


# ─────────────────────────────────────────────────────────────────────────────
# ENV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def create_env(OBELIX, args, seed: int, wall_obstacles: bool | None = None):
    use_walls = args.wall_obstacles if wall_obstacles is None else wall_obstacles
    return OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=use_walls,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK: Policy-over-Options  π(o | s)
# ─────────────────────────────────────────────────────────────────────────────

class OptionPolicy(nn.Module):
    """
    π(o | s) — decides which option (mode) to pursue.
    Input: normalised 18-dim raw obs → Categorical over N_OPTIONS.
    """

    def __init__(
        self,
        state_dim:   int   = N_OBS,
        n_options:   int   = N_OPTIONS,
        hidden:      int   = 64,
        temperature: float = 1.5,
        prob_floor:  float = 0.05,
    ):
        super().__init__()
        self.temperature = temperature
        self.prob_floor  = prob_floor
        self.n_options   = n_options

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_options),
        )
        nn.init.zeros_(self.net[-1].bias)   # uniform init

    def forward(
        self,
        x:       torch.Tensor,
        options: torch.Tensor | None = None,
    ):
        """
        Args:
            x:       (B, state_dim)
            options: optional LongTensor of taken options for log-prob recompute

        Returns: (option_sample, log_prob, entropy, probs)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        logits = self.net(x) / self.temperature
        probs  = torch.softmax(logits, dim=-1)
        probs  = (probs + self.prob_floor) / (1.0 + self.prob_floor * self.n_options)
        dist   = Categorical(probs=probs)
        o      = dist.sample() if options is None else options
        return o, dist.log_prob(o), dist.entropy(), probs


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK: Termination Function  β(o | s)
# ─────────────────────────────────────────────────────────────────────────────

class BetaNetwork(nn.Module):
    """
    β(o | s) — probability of terminating option o at state s.
    Outputs are clipped to [BETA_MIN, BETA_MAX] to prevent degeneracy.
    Init bias pushes sigmoid to ~0.12 so options don't terminate immediately.
    """

    BETA_MIN = 0.05
    BETA_MAX = 0.95

    def __init__(
        self,
        state_dim: int = N_OBS,
        n_options: int = N_OPTIONS,
        hidden:    int = 64,
    ):
        super().__init__()
        self.n_options = n_options
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_options),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns β ∈ [BETA_MIN, BETA_MAX]  —  shape (B, n_options)."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        raw  = torch.sigmoid(self.net(x))
        beta = raw * (self.BETA_MAX - self.BETA_MIN) + self.BETA_MIN
        return beta

    def termination_prob(self, x: torch.Tensor, option: int) -> torch.Tensor:
        """Scalar β for a single option  —  shape (B,)."""
        return self.forward(x)[:, option]


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK: Option-Value Function  Q(s, o)
# ─────────────────────────────────────────────────────────────────────────────

class OptionValueNetwork(nn.Module):
    """
    Q(s, o) — expected discounted return for option o from state s.
    V(s) = Σ_o π(o|s) · Q(s,o)  is derived on the fly where needed.
    """

    def __init__(
        self,
        state_dim: int = N_OBS,
        n_options: int = N_OPTIONS,
        hidden:    int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_options),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q(s, ·) — shape (B, n_options)."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)

    def q_for_option(self, x: torch.Tensor, option: int) -> torch.Tensor:
        return self.forward(x)[:, option]


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK: Intra-Option Policy  π(a | s, o)  — LSTM
# ─────────────────────────────────────────────────────────────────────────────

class IntraOptionPolicy(nn.Module):
    """
    π(a | s, o) — selects primitive actions conditioned on the active option.
    Structurally identical to the original LowLevelPolicy; option enters
    via the one-hot appended to the observation.
    """

    def __init__(
        self,
        state_dim:   int   = N_OBS_AUG,
        n_actions:   int   = N_ACTIONS,
        hidden:      int   = 128,
        temperature: float = 2.0,
        prob_floor:  float = 0.05,
        fw_floor:    float = 0.10,
    ):
        super().__init__()
        self.temperature = temperature
        self.prob_floor  = prob_floor
        self.fw_floor    = fw_floor
        self.n_actions   = n_actions

        self.fc   = nn.Linear(state_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out  = nn.Linear(hidden, n_actions)

        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[FW_IDX] = 1.0

        self.register_buffer("option_logit_bias", _OPTION_LOGIT_BIAS.clone())

    def forward(
        self,
        x:       torch.Tensor,
        hidden:  tuple | None        = None,
        actions: torch.Tensor | None = None,
    ):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        feat, hidden = self.lstm(torch.relu(self.fc(x)), hidden)
        logits       = self.out(feat)

        # Option-conditioned soft bias (non-trainable warm-start)
        opt_onehot = x[..., N_OBS:]
        opt_bias   = torch.einsum("...m,mn->...n", opt_onehot, self.option_logit_bias)
        logits     = (logits + opt_bias) / self.temperature

        # Global probability floor
        probs = torch.softmax(logits, dim=-1)
        probs = (probs + self.prob_floor) / (1.0 + self.prob_floor * self.n_actions)

        # FW-specific floor (no in-place ops to protect autograd)
        fw_floor_t = torch.full_like(probs[..., FW_IDX], self.fw_floor)
        fw_col     = torch.where(probs[..., FW_IDX] < self.fw_floor,
                                 fw_floor_t, probs[..., FW_IDX])
        mask              = torch.zeros_like(probs)
        mask[..., FW_IDX] = 1.0
        probs = probs * (1.0 - mask) + fw_col.unsqueeze(-1) * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = Categorical(probs=probs)
        a    = dist.sample() if actions is None else actions
        return a, dist.log_prob(a), dist.entropy(), probs.argmax(-1), hidden, probs


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _linear_decay(initial: float, final: float, ep: int, total: int) -> float:
    t = min(ep / max(total - 1, 1), 1.0)
    return initial * (1.0 - t) + final * t


def compute_gae(
    rewards: list[float],
    values:  list[float],
    gamma:   float = 0.99,
    lam:     float = 0.95,
) -> tuple[list[float], list[float]]:
    adv  = []
    gae  = 0.0
    vals = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t + 1] - vals[t]
        gae   = delta + gamma * lam * gae
        adv.insert(0, gae)
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


# ─────────────────────────────────────────────────────────────────────────────
# OPTION-CRITIC AGENT
# ─────────────────────────────────────────────────────────────────────────────

class OptionCriticAgent:
    """
    Option-Critic agent with four end-to-end learnable components:

      1. OptionPolicy      π(o|s)    which option to pursue
      2. IntraOptionPolicy π(a|s,o)  which action inside the option
      3. BetaNetwork       β(o|s)    when to terminate the option
      4. OptionValueNetwork Q(s,o)   value of option o in state s

    Termination is state-dependent and learned — no fixed duration K.

    Learning targets:
      A. Intra-option policy → PPO clipped surrogate
      B. Option policy       → PPO with A(s,o) = Q(s,o) - V(s)
      C. Termination         → β gradient: β(o|s)·(Q(s,o) - max_o' Q(s,o'))
      D. Q-network           → TD with option-conditional bootstrap
                               + EMA target network for stability
    """

    def __init__(self, env_fn, args):
        self.env_fn    = env_fn
        self.args      = args
        self.n_options = args.num_options
        self.gamma     = 0.99
        self.tau       = 0.005   # EMA coefficient for target Q-net

        self.rng = np.random.RandomState(args.seed)

        # ── Networks ──────────────────────────────────────────────────────────
        self.intra_policy = IntraOptionPolicy(
            state_dim=N_OBS_AUG,
            temperature=args.temperature,
            prob_floor=args.prob_floor,
            fw_floor=args.fw_floor,
        )
        self.option_policy = OptionPolicy(
            state_dim=N_OBS,
            n_options=self.n_options,
            temperature=args.hl_temperature,
            prob_floor=args.hl_prob_floor,
        )
        self.beta_net = BetaNetwork(
            state_dim=N_OBS,
            n_options=self.n_options,
        )
        self.q_net = OptionValueNetwork(
            state_dim=N_OBS,
            n_options=self.n_options,
        )
        # Target Q-network (EMA copy, frozen)
        self.q_target = OptionValueNetwork(
            state_dim=N_OBS,
            n_options=self.n_options,
        )
        self.q_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        # ── Optimisers ────────────────────────────────────────────────────────
        self.opt_intra  = optim.Adam(self.intra_policy.parameters(),  lr=args.lr_policy)
        self.opt_option = optim.Adam(self.option_policy.parameters(), lr=args.option_lr)
        self.opt_beta   = optim.Adam(self.beta_net.parameters(),      lr=args.beta_lr)
        self.opt_q      = optim.Adam(self.q_net.parameters(),         lr=args.lr_value)

        self.total_steps = 0

    # ── EMA target update ─────────────────────────────────────────────────────

    def _update_target(self) -> None:
        for p_on, p_tgt in zip(self.q_net.parameters(),
                                self.q_target.parameters()):
            p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p_on.data)

    # ── Exploration schedules ─────────────────────────────────────────────────

    def _ll_eps(self, ep: int) -> float:
        return _linear_decay(self.args.eps_start, self.args.eps_end,
                             ep, self.args.total_episodes)

    def _hl_eps(self, ep: int) -> float:
        return _linear_decay(self.args.hl_eps_start, self.args.hl_eps_end,
                             ep, self.args.total_episodes)

    # ── Option selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _pick_option(self, raw_obs: np.ndarray, ep: int) -> tuple[int, float]:
        """Sample a new option; return (option_idx, log_prob_under_π)."""
        if self.rng.rand() < self._hl_eps(ep):
            o = int(self.rng.randint(self.n_options))
        else:
            x = torch.tensor(prep_raw(raw_obs), dtype=torch.float32).unsqueeze(0)
            o_t, _, _, _ = self.option_policy(x)
            o = int(o_t.item())
        # Always recompute log_prob from the current π for accurate bookkeeping
        x    = torch.tensor(prep_raw(raw_obs), dtype=torch.float32).unsqueeze(0)
        _, lp, _, _ = self.option_policy(x, options=torch.tensor([o]))
        return o, float(lp.item())

    @torch.no_grad()
    def _should_terminate(self, raw_obs: np.ndarray, option: int) -> bool:
        """Sample termination from β(option | s)."""
        x    = torch.tensor(prep_raw(raw_obs), dtype=torch.float32).unsqueeze(0)
        beta = self.beta_net.termination_prob(x, option)
        return bool(Bernoulli(beta).sample().item())

    # ── Rollout ───────────────────────────────────────────────────────────────

    def rollout(self, env, ep: int = 0, render: bool = False) -> dict:
        """
        One episode under the Option-Critic framework.

        At every step:
          1. Check β(current_option | s)  →  terminate?
          2. If terminating OR first step  →  sample new option from π(o|s)
          3. Augment s with option one-hot
          4. Sample primitive action from π(a | s, o)

        Returns a trajectory dict with per-step arrays.
        """
        raw_obs = np.asarray(env.reset(), dtype=np.float32)
        done    = False
        ll_eps  = self._ll_eps(ep)

        traj = dict(
            raw_obs      = [],   # normalised 18-dim, for option-level nets
            aug_obs      = [],   # normalised 22-dim, for intra-option net
            actions      = [],
            options      = [],
            rewards      = [],
            intra_logps  = [],   # log π(a|s,o)
            option_logps = [],   # log π(o|s)  at selection time
            term_flags   = [],   # 1 if option was freshly selected this step
            beta_vals    = [],   # β(current_option | s)  for logging
        )

        h_p: tuple | None = None   # LSTM hidden state

        # First step: always pick an option
        current_option, opt_logp = self._pick_option(raw_obs, ep)
        fresh_option             = True

        while not done:
            # ── Termination check (not on the very first step) ────────────────
            if not fresh_option:
                if self._should_terminate(raw_obs, current_option):
                    current_option, opt_logp = self._pick_option(raw_obs, ep)
                    h_p          = None     # reset LSTM memory on option switch
                    fresh_option = True
                # else: fresh_option stays False
            else:
                fresh_option = False   # will be False for all subsequent steps

            # ── Option-level scalar quantities (no grad) ──────────────────────
            x_raw = torch.tensor(prep_raw(raw_obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                beta_o = float(
                    self.beta_net.termination_prob(x_raw, current_option).item()
                )

            # ── Intra-option action selection ─────────────────────────────────
            s   = prep_obs(raw_obs, current_option)
            s_t = torch.tensor(s, dtype=torch.float32).view(1, 1, -1)

            with torch.no_grad():
                a_net, logp, _, _, h_p, _ = self.intra_policy(s_t, h_p)

            h_p = (h_p[0].detach(), h_p[1].detach())

            if self.rng.rand() < ll_eps:
                a = int(self.rng.randint(N_ACTIONS))
            else:
                a = int(a_net.item())

            self.total_steps += 1

            raw_obs2, r, done = env.step(ACTIONS[a], render=render)
            raw_obs2 = np.asarray(raw_obs2, dtype=np.float32)

            if a in ROT_IDX:
                r -= self.args.rotation_penalty

            # ── Store ─────────────────────────────────────────────────────────
            traj["raw_obs"].append(prep_raw(raw_obs).astype(np.float32))
            traj["aug_obs"].append(s)
            traj["actions"].append(a)
            traj["options"].append(current_option)
            traj["rewards"].append(float(r))
            traj["intra_logps"].append(float(logp.item()))
            traj["option_logps"].append(opt_logp)
            traj["term_flags"].append(int(fresh_option))
            traj["beta_vals"].append(beta_o)

            raw_obs = raw_obs2

        return traj

    # ── TD targets for Q(s, o) ────────────────────────────────────────────────

    def _compute_q_targets(self, traj: dict) -> torch.Tensor:
        """
        Option-conditional TD target (using frozen target network):

          y_t = r_t + γ · [(1 - β(o|s_{t+1})) · Q_tgt(s_{t+1}, o)
                          +     β(o|s_{t+1})   · max_o' Q_tgt(s_{t+1}, o')]

        The last step bootstraps to 0 (episode ended).
        """
        T       = len(traj["rewards"])
        targets = []
        raw_arr = traj["raw_obs"]

        for t in range(T):
            r = traj["rewards"][t]
            o = traj["options"][t]

            if t + 1 < T:
                s_next = raw_arr[t + 1]
            else:
                # Terminal: no bootstrap
                targets.append(r)
                continue

            x_next = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_next    = self.q_target(x_next).squeeze(0)
                beta_next = float(
                    self.beta_net.termination_prob(x_next, o).item()
                )
            v_next = (
                (1.0 - beta_next) * q_next[o].item()
                + beta_next * q_next.max().item()
            )
            targets.append(r + self.gamma * v_next)

        return torch.tensor(targets, dtype=torch.float32)

    # ── Update A: intra-option policy (PPO) ──────────────────────────────────

    def _update_intra_option(
        self,
        aug_obs_t:   torch.Tensor,
        actions_t:   torch.Tensor,
        old_logps_t: torch.Tensor,
        adv_t:       torch.Tensor,
        ent_coef:    float,
        n_epochs:    int = 10,
    ) -> None:
        for _ in range(n_epochs):
            _, logp, ent, _, _, probs_b = self.intra_policy(
                aug_obs_t, actions=actions_t.unsqueeze(1)
            )
            logp = logp.squeeze(1)

            ratio = torch.exp(logp - old_logps_t)
            surr  = torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 0.8, 1.2) * adv_t,
            )
            # Diversity regulariser (discourages action collapse)
            action_freq = probs_b.squeeze(1).mean(0)
            uniform     = torch.ones_like(action_freq) / N_ACTIONS
            div_loss    = ((action_freq - uniform) ** 2).mean()

            kl     = (old_logps_t - logp).mean()
            loss_p = -surr.mean() - ent_coef * ent.mean() + 0.5 * kl + 0.1 * div_loss

            self.opt_intra.zero_grad()
            loss_p.backward()
            nn.utils.clip_grad_norm_(self.intra_policy.parameters(), 0.5)
            self.opt_intra.step()

    # ── Update B: option policy (PPO) ────────────────────────────────────────

    def _update_option_policy(
        self,
        raw_obs_t:   torch.Tensor,
        options_t:   torch.Tensor,
        old_logps_t: torch.Tensor,
        adv_t:       torch.Tensor,
        ent_coef:    float,
        n_epochs:    int = 5,
    ) -> None:
        """PPO update for π(o|s).  Advantage: A(s,o) = Q(s,o) - V(s)."""
        for _ in range(n_epochs):
            _, logp, ent, _ = self.option_policy(raw_obs_t, options=options_t)

            ratio = torch.exp(logp - old_logps_t)
            surr  = torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 0.8, 1.2) * adv_t,
            )
            kl     = (old_logps_t - logp).mean()
            loss_o = -surr.mean() - ent_coef * ent.mean() + 0.5 * kl

            self.opt_option.zero_grad()
            loss_o.backward()
            nn.utils.clip_grad_norm_(self.option_policy.parameters(), 0.5)
            self.opt_option.step()

    # ── Update C: termination function ───────────────────────────────────────

    def _update_termination(
        self,
        raw_obs_t: torch.Tensor,
        options_t: torch.Tensor,
        term_mask: torch.Tensor,   # 1 = option just switched, 0 = still running
    ) -> None:
        """
        Termination gradient:
          L_β = Σ_t  β(o_t|s_t) · (Q(s_t,o_t) - max_o' Q(s_t,o'))

        Positive gap  →  current option is already best  →  push β down.
        Negative gap  →  better option exists           →  push β up.

        We mask out freshly-switched steps (term_mask == 1) because at
        those steps the old option has already terminated; gradient there
        would be spurious.
        """
        with torch.no_grad():
            q_all = self.q_net(raw_obs_t)            # (T, N_OPTIONS)

        q_cur  = q_all.gather(1, options_t.unsqueeze(1)).squeeze(1)
        q_best = q_all.max(dim=1).values

        # advantage_gap > 0 means current option is competitive — don't terminate
        advantage_gap = (q_cur - q_best).detach()

        # Only penalise steps where the option is still running
        non_switch = (term_mask == 0).float()

        beta_all = self.beta_net(raw_obs_t)           # (T, N_OPTIONS)
        beta_cur = beta_all.gather(1, options_t.unsqueeze(1)).squeeze(1)

        loss_beta = (beta_cur * advantage_gap * non_switch).mean()

        self.opt_beta.zero_grad()
        loss_beta.backward()
        nn.utils.clip_grad_norm_(self.beta_net.parameters(), 0.5)
        self.opt_beta.step()

    # ── Update D: Q-network (TD) ──────────────────────────────────────────────

    def _update_q(
        self,
        raw_obs_t: torch.Tensor,
        options_t: torch.Tensor,
        q_targets: torch.Tensor,
    ) -> None:
        q_all = self.q_net(raw_obs_t)
        q_cur = q_all.gather(1, options_t.unsqueeze(1)).squeeze(1)
        loss_q = F.mse_loss(q_cur, q_targets)

        self.opt_q.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.opt_q.step()

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        args           = self.args
        total_episodes = args.total_episodes
        base_seed      = args.seed
        wall_switch    = args.wall_switch

        print("=" * 72)
        print("  Option-Critic PPO + LSTM  |  OBELIX")
        print("=" * 72)
        cfg = dict(
            seed             = base_seed,
            total_episodes   = total_episodes,
            num_options      = self.n_options,
            lr_policy        = args.lr_policy,
            lr_value         = args.lr_value,
            option_lr        = args.option_lr,
            beta_lr          = args.beta_lr,
            temperature      = args.temperature,
            hl_temperature   = args.hl_temperature,
            prob_floor       = args.prob_floor,
            hl_prob_floor    = args.hl_prob_floor,
            fw_floor         = args.fw_floor,
            eps_start        = args.eps_start,
            eps_end          = args.eps_end,
            hl_eps_start     = args.hl_eps_start,
            hl_eps_end       = args.hl_eps_end,
            rotation_penalty = args.rotation_penalty,
            wall_switch      = wall_switch,
        )
        for k, v in cfg.items():
            print(f"  {k:<24} = {v}")
        print("=" * 72, flush=True)

        for ep in range(total_episodes):
            use_walls      = ep >= wall_switch
            render_this_ep = ep % args.eval_every == 0

            env_seed = base_seed + ep
            env      = self.env_fn(env_seed, wall_obstacles=use_walls)
            phase    = "WITH walls" if use_walls else "no walls "

            if ep == wall_switch:
                print("=" * 72)
                print(f"  CURRICULUM: wall_obstacles=True starts at ep {ep}")
                print("=" * 72, flush=True)

            traj      = self.rollout(env, ep=ep, render=render_this_ep)
            T         = len(traj["rewards"])
            ep_return = sum(traj["rewards"])

            if render_this_ep:
                print(f"  [render ep {ep:4d} | {phase}]  return={ep_return:.2f}",
                      flush=True)

            # ── Build tensors ─────────────────────────────────────────────────
            raw_obs_t  = torch.from_numpy(
                np.array(traj["raw_obs"], dtype=np.float32)
            )                                          # (T, 18)
            aug_obs_t  = torch.from_numpy(
                np.array(traj["aug_obs"], dtype=np.float32)
            ).unsqueeze(1)                             # (T, 1, 22)

            actions_t   = torch.tensor(traj["actions"],      dtype=torch.long)
            options_t   = torch.tensor(traj["options"],      dtype=torch.long)
            term_mask   = torch.tensor(traj["term_flags"],   dtype=torch.float32)
            intra_lp_t  = torch.tensor(traj["intra_logps"],  dtype=torch.float32)
            option_lp_t = torch.tensor(traj["option_logps"], dtype=torch.float32)

            # ── D: Q-network update ───────────────────────────────────────────
            q_targets = self._compute_q_targets(traj)
            self._update_q(raw_obs_t, options_t, q_targets)
            self._update_target()

            # ── A: intra-option advantage  (Q_target - Q_online as baseline) ──
            with torch.no_grad():
                q_cur_t = self.q_net(raw_obs_t).gather(
                    1, options_t.unsqueeze(1)
                ).squeeze(1)
            intra_adv = q_targets - q_cur_t
            intra_adv = (intra_adv - intra_adv.mean()) / (intra_adv.std() + 1e-8)
            intra_adv = torch.clamp(intra_adv, -2.0, 5.0)
            if ep < 100:
                intra_adv = torch.clamp(intra_adv, min=-1.0)

            # ── B: option-policy advantage  A(s,o) = Q(s,o) - V(s) ───────────
            with torch.no_grad():
                q_all_t    = self.q_net(raw_obs_t)
                _, _, _, opt_probs = self.option_policy(raw_obs_t)
                v_s        = (q_all_t * opt_probs).sum(dim=-1)
                q_o        = q_all_t.gather(1, options_t.unsqueeze(1)).squeeze(1)
            option_adv = (q_o - v_s).detach()
            option_adv = (option_adv - option_adv.mean()) / (option_adv.std() + 1e-8)

            # ── Annealed entropy coefficients ─────────────────────────────────
            ll_ent = max(0.02, 0.10 * (1.0 - ep / total_episodes))
            hl_ent = max(0.05, 0.20 * (1.0 - ep / total_episodes))

            # ── Run A, B, C ───────────────────────────────────────────────────
            self._update_intra_option(
                aug_obs_t, actions_t, intra_lp_t, intra_adv, ll_ent
            )
            self._update_option_policy(
                raw_obs_t, options_t, option_lp_t, option_adv, hl_ent
            )
            self._update_termination(raw_obs_t, options_t, term_mask)

            # ── Logging ───────────────────────────────────────────────────────
            option_counts = [0] * self.n_options
            for o in traj["options"]:
                option_counts[o] += 1
            opt_str = "  OPT[" + " ".join(
                f"{OPTION_NAMES[i][0]}:{100*option_counts[i]/max(T,1):.0f}%"
                for i in range(self.n_options)
            ) + "]"

            avg_beta = float(np.mean(traj["beta_vals"]))
            n_switches = int(sum(traj["term_flags"]))

            with torch.no_grad():
                n_dbg = min(8, aug_obs_t.shape[0])
                _, _, ent_dbg, _, _, probs_dbg = self.intra_policy(
                    aug_obs_t[:n_dbg],
                    actions=actions_t[:n_dbg].unsqueeze(1),
                )
                mean_probs = probs_dbg.squeeze(1).mean(0)

            act_str = "  ".join(
                f"{ACTIONS[i]}:{mean_probs[i].item()*100:.1f}%"
                for i in range(N_ACTIONS)
            )
            fw_pct = mean_probs[FW_IDX].item() * 100
            flag   = "  ◀ FW LOW!" if fw_pct < 10 else ""

            print(
                f"Ep {ep:4d} | {phase} | ret={ep_return:+8.1f} | "
                f"ent={ent_dbg.mean().item():.3f} | "
                f"β_avg={avg_beta:.3f} sw={n_switches:3d} | "
                f"ll_eps={self._ll_eps(ep):.3f} hl_eps={self._hl_eps(ep):.3f} | "
                f"steps={self.total_steps} | "
                f"[{act_str}]{opt_str}{flag}"
            )

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {
                "intra_policy":  self.intra_policy.state_dict(),
                "option_policy": self.option_policy.state_dict(),
                "beta_net":      self.beta_net.state_dict(),
                "q_net":         self.q_net.state_dict(),
                "q_target":      self.q_target.state_dict(),
            },
            path,
        )
        print(f"\nSaved Option-Critic checkpoint → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.intra_policy.load_state_dict(ckpt["intra_policy"])
        self.option_policy.load_state_dict(ckpt["option_policy"])
        self.beta_net.load_state_dict(ckpt["beta_net"])
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        print(f"Loaded Option-Critic checkpoint ← {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument("--obelix_py",      type=str, required=True)
    parser.add_argument("--out",            type=str, default="weights_option_critic.pth")
    parser.add_argument("--load",           type=str, default=None)

    # ── Reproducibility ───────────────────────────────────────────────────────
    parser.add_argument("--seed",           type=int, default=42)

    # ── Option-Critic specific ────────────────────────────────────────────────
    parser.add_argument("--num_options",    type=int,   default=N_OPTIONS,
                        help="Number of learnable options")
    parser.add_argument("--option_lr",      type=float, default=5e-5,
                        help="Learning rate for π(o|s)")
    parser.add_argument("--beta_lr",        type=float, default=5e-5,
                        help="Learning rate for β(o|s)")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--total_episodes", type=int,   default=1000)
    parser.add_argument("--eval_every",     type=int,   default=10)

    # ── Intra-option network ──────────────────────────────────────────────────
    parser.add_argument("--temperature",    type=float, default=2.0)
    parser.add_argument("--prob_floor",     type=float, default=0.05)
    parser.add_argument("--fw_floor",       type=float, default=0.20)

    # ── Option-policy network ─────────────────────────────────────────────────
    parser.add_argument("--hl_temperature", type=float, default=1.2)
    parser.add_argument("--hl_prob_floor",  type=float, default=0.05)

    # ── Low-level exploration ─────────────────────────────────────────────────
    parser.add_argument("--eps_start",      type=float, default=0.10)
    parser.add_argument("--eps_end",        type=float, default=0.01)

    # ── Option-level exploration ──────────────────────────────────────────────
    parser.add_argument("--hl_eps_start",   type=float, default=0.30)
    parser.add_argument("--hl_eps_end",     type=float, default=0.05)

    # ── Penalties ─────────────────────────────────────────────────────────────
    parser.add_argument("--rotation_penalty", type=float, default=0.1)

    # ── Optimisation ──────────────────────────────────────────────────────────
    parser.add_argument("--lr_policy",      type=float, default=1e-4,
                        help="Intra-option policy LR")
    parser.add_argument("--lr_value",       type=float, default=1e-3,
                        help="Q-network LR")

    # ── Environment ───────────────────────────────────────────────────────────
    parser.add_argument("--difficulty",     type=int,   default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--wall_switch",    type=int,   default=300)
    parser.add_argument("--box_speed",      type=int,   default=2)
    parser.add_argument("--scaling_factor", type=int,   default=5)
    parser.add_argument("--arena_size",     type=int,   default=500)
    parser.add_argument("--max_steps",      type=int,   default=200)

    args = parser.parse_args()
    set_global_seeds(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    def env_fn(seed: int, wall_obstacles: bool | None = None):
        return create_env(OBELIX, args, seed, wall_obstacles=wall_obstacles)

    agent = OptionCriticAgent(env_fn, args)
    if args.load:
        agent.load(args.load)

    agent.train()
    agent.save(args.out)


if __name__ == "__main__":
    main()