"""PPO with World Model for OBELIX — RSSM-style belief state.

Trains a world model (LSTM) to predict next observations from (obs_t, action_t),
then uses the hidden state z_t as belief state input to PPO instead of raw obs.

This converts the POMDP to a near-MDP by maintaining belief about box location
even when sonar is all zeros.

Run:
  python train_rssm_ppo.py --obelix_py ./obelix.py --out weights_rssm_ppo.pth
"""

from __future__ import annotations

import argparse
import os
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

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_OBS     = 18
N_ACTIONS = 5

MAX_POLICY_GRAD = 0.5
MAX_VALUE_GRAD  = 0.5


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
#  WorldModel class
# ─────────────────────────────────────────────────────────────────────────────
class WorldModel(nn.Module):
    def __init__(self, obs_dim=18, action_dim=5, hidden_dim=64, z_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        # Encoder: obs_t + action_t → latent
        self.encoder = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Prediction head: predict next obs (binary)
        self.pred_head = nn.Linear(hidden_dim, obs_dim)
        # Encode head: compress to belief state z_t
        self.encode_head = nn.Linear(hidden_dim, z_dim)

    def forward(self, obs_seq, act_seq, hx=None):
        # obs_seq : (B, T, obs_dim)  float32
        # act_seq : (B, T, action_dim)  one-hot float32
        # hx      : LSTM hidden state or None
        # returns : pred_obs (B,T,obs_dim), z (B,T,z_dim), new_hx
        x = torch.cat([obs_seq, act_seq], dim=-1)  # (B, T, obs_dim + action_dim)
        encoded = F.relu(self.encoder(x))  # (B, T, hidden_dim)
        lstm_out, hx = self.lstm(encoded, hx)  # lstm_out: (B, T, hidden_dim)
        pred_obs = torch.sigmoid(self.pred_head(lstm_out))  # (B, T, obs_dim)
        z = self.encode_head(lstm_out)  # (B, T, z_dim)
        return pred_obs, z, hx

    def init_hx(self, batch_size=1):
        # returns zeroed (h, c) each shape (1, B, hidden_dim)
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)
        return (h, c)

    def encode_step(self, obs, action_idx, hx):
        # single step inference for rollout use
        # obs        : np.ndarray (obs_dim,)
        # action_idx : int
        # hx         : current hidden state (h, c)
        # returns    : z_t (np.ndarray, z_dim), new_hx
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
        act_onehot = torch.zeros(1, 1, self.action_dim)
        act_onehot[0, 0, action_idx] = 1.0
        _, z, hx = self.forward(obs_t, act_onehot, hx)
        return z.squeeze(0).squeeze(0).detach().numpy(), hx


# ─────────────────────────────────────────────────────────────────────────────
#  WorldModelTrainer class
# ─────────────────────────────────────────────────────────────────────────────
class WorldModelTrainer:
    def __init__(self, world_model, env_fn, args):
        self.world_model = world_model
        self.env_fn = env_fn
        self.args = args

    def collect_data(self, n_episodes=200):
        # Run random policy for n_episodes
        # Store every (obs_t, action_idx, obs_{t+1}) transition
        # Return as three arrays:
        #   obs_arr      : (N, obs_dim)  all obs_t
        #   actions_arr  : (N,)     all action indices
        #   next_obs_arr : (N, obs_dim)  all obs_{t+1}
        # IMPORTANT: also store episode boundaries so sequences
        # are not constructed across episode resets
        obs_list = []
        actions_list = []
        next_obs_list = []
        episode_starts = [0]  # indices where episodes start

        for ep in range(n_episodes):
            env = self.env_fn(seed=self.args.seed + 10000 + ep)
            obs = np.asarray(env.reset(), dtype=np.float32)
            done = False
            step = 0
            while not done and step < self.args.max_steps:
                action_idx = np.random.randint(0, N_ACTIONS)
                obs_next, _, done = env.step(ACTIONS[action_idx], render=False)
                obs_next = np.asarray(obs_next, dtype=np.float32)

                obs_list.append(obs)
                actions_list.append(action_idx)
                next_obs_list.append(obs_next)

                obs = obs_next
                step += 1

            episode_starts.append(len(obs_list))
            try:
                env.close()
            except Exception:
                pass

        obs_arr = np.array(obs_list, dtype=np.float32)
        actions_arr = np.array(actions_list, dtype=np.int64)
        next_obs_arr = np.array(next_obs_list, dtype=np.float32)

        # episode_starts includes 0 and cumulative lengths, remove last (total length)
        self.episode_starts = episode_starts[:-1]
        return obs_arr, actions_arr, next_obs_arr

    def train(self, n_epochs=50, batch_size=32, seq_len=20, lr=1e-3):
        # Sample random sequences of length seq_len from collected data
        # Respect episode boundaries — never stitch two episodes together
        # Each batch: obs (B,T,obs_dim), actions_onehot (B,T,action_dim), targets (B,T,obs_dim)
        # targets[t] = obs[t+1]  (next observation)
        # Feed full sequence through world_model.forward()
        # Compute binary_cross_entropy on pred_obs vs targets
        # Print loss every 10 epochs
        # Return trained world_model

        obs_arr, actions_arr, next_obs_arr = self.collect_data(self.args.wm_episodes)
        N = len(obs_arr)

        optimizer = optim.Adam(self.world_model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            # Sample sequences respecting episode boundaries
            episode_ends = self.episode_starts[1:] + [N]  # end indices for each episode
            sequences = []
            for ep_idx, start in enumerate(self.episode_starts):
                end = episode_ends[ep_idx]
                ep_len = end - start
                if ep_len >= seq_len + 1:  # need seq_len + 1 for targets
                    for i in range(ep_len - seq_len):
                        seq_start = start + i
                        seq_end = seq_start + seq_len
                        sequences.append((seq_start, seq_end))

            if not sequences:
                continue

            # Shuffle sequences
            np.random.shuffle(sequences)

            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, len(sequences), batch_size):
                batch_sequences = sequences[batch_start:batch_start + batch_size]
                if not batch_sequences:
                    continue

                # Build batch
                B = len(batch_sequences)
                obs_batch = np.zeros((B, seq_len, self.world_model.obs_dim), dtype=np.float32)
                act_batch = np.zeros((B, seq_len, self.world_model.action_dim), dtype=np.float32)
                target_batch = np.zeros((B, seq_len, self.world_model.obs_dim), dtype=np.float32)

                for b, (seq_start, seq_end) in enumerate(batch_sequences):
                    obs_batch[b] = obs_arr[seq_start:seq_end]
                    for t in range(seq_len):
                        act_idx = actions_arr[seq_start + t]
                        act_batch[b, t, act_idx] = 1.0
                    target_batch[b] = next_obs_arr[seq_start:seq_end]

                obs_t = torch.tensor(obs_batch)
                act_t = torch.tensor(act_batch)
                target_t = torch.tensor(target_batch)

                pred_obs, _, _ = self.world_model(obs_t, act_t)
                loss = F.binary_cross_entropy(pred_obs, target_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 10 == 0 and n_batches > 0:
                print(f"WorldModel epoch {epoch}: loss = {epoch_loss / n_batches:.4f}")

        return self.world_model

    def evaluate(self, n_episodes=10):
        # Run trained world model on fresh episodes with random policy
        # Print average next-obs prediction accuracy per bit
        # Print which sensor bits are hardest to predict
        # This tells you if the world model has learned box dynamics
        obs_arr, actions_arr, next_obs_arr = self.collect_data(n_episodes)
        N = len(obs_arr)

        # Evaluate in batches, but reset hx at episode boundaries
        batch_size = 64
        accuracies = np.zeros(self.world_model.obs_dim)

        self.world_model.eval()
        episode_ends = self.episode_starts[1:] + [N]
        
        with torch.no_grad():
            start_idx = 0
            hx = self.world_model.init_hx(1)
            for ep_idx, ep_start in enumerate(self.episode_starts):
                ep_end = episode_ends[ep_idx]
                # Reset hx at start of each episode
                hx = self.world_model.init_hx(1)
                
                for i in range(ep_start, ep_end):
                    obs_t = torch.tensor(obs_arr[i:i+1], dtype=torch.float32).unsqueeze(0)
                    act_onehot = torch.zeros(1, 1, self.world_model.action_dim)
                    act_onehot[0, 0, actions_arr[i]] = 1.0
                    pred, _, hx = self.world_model(obs_t, act_onehot, hx)
                    pred_obs = pred.squeeze(0).squeeze(0).numpy()
                    
                    # Binary accuracy per bit
                    pred_binary = (pred_obs > 0.5).astype(np.float32)
                    correct = (pred_binary == next_obs_arr[i]).astype(np.float32)
                    accuracies += correct

        accuracies /= N
        print("WorldModel evaluation - prediction accuracies per bit:")
        for i in range(self.world_model.obs_dim):
            print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")


# ─────────────────────────────────────────────────────────────────────────────
#  Modified PPO Networks (input is z_dim, not obs_dim)
# ─────────────────────────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    """inputLayer → hLayers → out, activation = F.relu."""

    def __init__(self, stateDim=32, n_actions=N_ACTIONS,
                 hDims=(64, 64), activationFn=F.relu):
        super().__init__()
        self.activation = activationFn
        self.inputLayer = nn.Linear(stateDim, hDims[0])
        self.hLayers    = nn.ModuleList(
            [nn.Linear(hDims[i], hDims[i + 1]) for i in range(len(hDims) - 1)]
        )
        self.out = nn.Linear(hDims[-1], n_actions)

    def forward(self, states, actions=None):
        """Returns (sampled_actions, log_probs, entropies, greedy_actions).

        If actions is None, samples from the distribution.
        If actions provided, evaluates log_prob for those actions.
        """
        ss = states if isinstance(states, torch.Tensor) \
             else torch.tensor(states, dtype=torch.float32)
        l = self.activation(self.inputLayer(ss))
        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        logits = self.out(l)

        dist         = Categorical(logits=logits)
        greedy       = logits.argmax(dim=-1)

        if actions is None:
            a = dist.sample()
        else:
            a = actions if isinstance(actions, torch.Tensor) \
                else torch.tensor(actions, dtype=torch.long)

        return a, dist.log_prob(a), dist.entropy(), greedy


class ValueNetwork(nn.Module):
    """inputLayer → hLayers → out (scalar V)."""

    def __init__(self, stateDim=32, hDims=(64, 64), activationFn=F.relu):
        super().__init__()
        self.activation = activationFn
        self.inputLayer = nn.Linear(stateDim, hDims[0])
        self.hLayers    = nn.ModuleList(
            [nn.Linear(hDims[i], hDims[i + 1]) for i in range(len(hDims) - 1)]
        )
        self.out = nn.Linear(hDims[-1], 1)

    def forward(self, states):
        ss = states if isinstance(states, torch.Tensor) \
             else torch.tensor(states, dtype=torch.float32)
        l = self.activation(self.inputLayer(ss))
        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        q = self.out(l)
        return q   # shape (..., 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Modified EpisodeBuffer (stores z_t instead of obs)
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeBuffer:
    def __init__(self, gamma: float, lam: float,
                 stateDim: int, numWorkers: int,
                 maxEpisodes: int, maxEpisodeSteps: int):
        self.gamma           = gamma
        self.lam             = lam
        self.stateDim        = stateDim  # now z_dim
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

    def fill(self, envs, world_model, pNetwork, vNetwork):
        """Collect episodes from worker envs using world model for belief states."""
        numWorkers = self.numWorkers
        T          = self.maxEpisodeSteps

        # Initialize world model hidden states for each worker
        wm_hxs = [world_model.init_hx(1) for _ in range(numWorkers)]

        ss = np.stack([np.asarray(env.reset(), dtype=np.float32)
                       for env in envs])

        # Get initial z_t (use FW action as prev_action for t=0)
        zs = []
        for i in range(numWorkers):
            z_t, wm_hxs[i] = world_model.encode_step(ss[i], 2, wm_hxs[i])  # FW = 2
            zs.append(z_t)
        zs = np.array(zs)

        bufferFull = False
        wRewards   = np.zeros((numWorkers, T), dtype=np.float32)
        wSteps     = np.zeros(numWorkers, dtype=np.int64)

        pNetwork.eval(); vNetwork.eval()
        world_model.eval()

        while not bufferFull:
            with torch.no_grad():
                zs_t = torch.tensor(zs, dtype=torch.float32)
                actions_t, logp_t, _, _ = pNetwork.forward(zs_t)
                vs_t = vNetwork.forward(zs_t).squeeze(-1)

            actions = actions_t.numpy()
            logp_as = logp_t.numpy()

            for id_ in range(numWorkers):
                epID = self.currentEpisodeIDs[id_]
                t    = wSteps[id_]
                self.bufferStates [epID, t] = zs[id_]
                self.bufferActions[epID, t] = actions[id_]
                self.bufferLogp_as[epID, t] = logp_as[id_]

            sNexts = np.zeros_like(ss)
            rs     = np.zeros(numWorkers, dtype=np.float32)
            dones  = np.zeros(numWorkers, dtype=np.int32)

            for id_ in range(numWorkers):
                obs, r, done = envs[id_].step(ACTIONS[actions[id_]], render=False)
                sNexts[id_] = np.asarray(obs, dtype=np.float32)
                rs[id_]     = float(r)
                dones[id_]  = int(done)

            for id_ in range(numWorkers):
                wRewards[id_, wSteps[id_]] = rs[id_]

            for id_ in range(numWorkers):
                if wSteps[id_] + 1 == T:
                    dones[id_] = 1

            if dones.sum():
                dones_ids = np.where(dones)[0]
                nValues   = np.zeros(numWorkers, dtype=np.float32)
                with torch.no_grad():
                    # Get z_t for next states
                    z_nexts = []
                    for id_ in dones_ids:
                        z_t, _ = world_model.encode_step(sNexts[id_], actions[id_], wm_hxs[id_])
                        z_nexts.append(z_t)
                    z_nexts = np.array(z_nexts)
                    zn_t = torch.tensor(z_nexts, dtype=torch.float32)
                    nValues[dones_ids] = vNetwork.forward(zn_t).squeeze(-1).numpy()

                for id_ in dones_ids:
                    sNexts[id_] = np.asarray(envs[id_].reset(), dtype=np.float32)
                    # Reset world model hx for new episode
                    wm_hxs[id_] = world_model.init_hx(1)
                    # Get initial z_t for reset state
                    z_t, wm_hxs[id_] = world_model.encode_step(sNexts[id_], 2, wm_hxs[id_])  # FW

                for id_ in dones_ids:
                    epID = self.currentEpisodeIDs[id_]
                    ep_T = int(wSteps[id_]) + 1
                    self.episodeSteps[epID]   = ep_T
                    self.episodeRewards[epID] = float(wRewards[id_, :ep_T].sum())

                    # Discounted returns
                    epRewards   = np.append(wRewards[id_, :ep_T], nValues[id_])
                    epDiscounts = self.discounts[:ep_T + 1]
                    epReturns   = [np.sum(epDiscounts[:ep_T + 1 - t] * epRewards[t:])
                                   for t in range(ep_T)]
                    self.bufferReturns[epID, :ep_T] = np.array(epReturns, dtype=np.float32)

                    # GAE
                    epZs = self.bufferStates[epID, :ep_T]
                    with torch.no_grad():
                        epV = vNetwork.forward(
                            torch.tensor(epZs, dtype=torch.float32)
                        ).squeeze(-1).numpy()
                    epValues = np.append(epV, nValues[id_])
                    epTau    = self.tau[:ep_T]
                    deltas   = epRewards[:-1] + self.gamma * epValues[1:] - epValues[:-1]
                    gaes     = [np.sum(epTau[:ep_T - t] * deltas[t:]) for t in range(ep_T)]
                    self.bufferGAEs[epID, :ep_T] = np.array(gaes, dtype=np.float32)

                    new_epID = max(self.currentEpisodeIDs) + 1
                    if new_epID >= self.maxEpisodes:
                        bufferFull = True
                        break
                    self.currentEpisodeIDs[id_] = new_epID
                    wRewards[id_] = 0.0
                    wSteps[id_]   = -1

            # Update zs for next step
            z_nexts = []
            for id_ in range(numWorkers):
                if dones[id_]:
                    z_nexts.append(zs[id_])  # already updated above
                else:
                    z_t, wm_hxs[id_] = world_model.encode_step(sNexts[id_], actions[id_], wm_hxs[id_])
                    z_nexts.append(z_t)
            zs = np.array(z_nexts)

            ss      = sNexts
            wSteps += 1

        pNetwork.train(); vNetwork.train()
        world_model.train()


# ─────────────────────────────────────────────────────────────────────────────
#  Modified PPO class
# ─────────────────────────────────────────────────────────────────────────────
class PPO:
    def __init__(
        self,
        env,                        # single OBELIX env for eval
        world_model,                # trained world model
        gamma:                float,
        lam:                  float,
        beta:                 float,   # entropy coefficient
        numWorkers:           int,
        maxEpisodes:          int,
        maxEpisodeSteps:      int,
        updateFrequency:      int,     # not used when buffer drives training
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
        # network architecture
        z_dim:                int   = 32,
        n_actions:            int   = N_ACTIONS,
        hDims_policy:         tuple = (64, 64),
        hDims_value:          tuple = (64, 64),
        # training budget
        MAX_TRAIN_EPISODES:   int   = 1000,
        seed:                 int   = 42,
        eval_render:          bool  = False,
        eval_env_fn                 = None,
    ):
        self.env                       = env
        self.world_model               = world_model
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

        self.pNetwork = PolicyNetwork(z_dim, n_actions, hDims_policy)
        self.vNetwork = ValueNetwork(z_dim, hDims_value)

        self.policyOptimizerFn = policyOptimizerFn(
            self.pNetwork.parameters(), lr=policyOptimizerLR)
        self.valueOptimizerFn  = valueOptimizerFn(
            self.vNetwork.parameters(), lr=valueOptimizerLR)

        self.rBuffer = EpisodeBuffer(
            gamma=gamma, lam=lam,
            stateDim=z_dim, numWorkers=numWorkers,
            maxEpisodes=maxEpisodes, maxEpisodeSteps=maxEpisodeSteps,
        )

        self.initBookKeeping()

    # ── initBookKeeping ───────────────────────────────────────────────────────
    def initBookKeeping(self):
        self.trainEpisodeRewards:  list[float] = []
        self.evalEpisodeRewards:   list[float] = []
        self.policyLosses:         list[float] = []
        self.valueLosses:          list[float] = []
        self.entropies:            list[float] = []
        self.wallTimes:            list[float] = []
        self._t0 = time.perf_counter()

    # ── runPPO ────────────────────────────────────────────────────────────────
    def runPPO(self):
        resultTrain = self.trainAgent()
        evalMean, evalStd = self.evaluateAgent()
        trainingTime  = time.perf_counter() - self._t0
        wallclockTime = self.wallTimes[-1] if self.wallTimes else 0.0
        return resultTrain, evalMean, trainingTime, wallclockTime

    # ── trainAgent ────────────────────────────────────────────────────────────
    def trainAgent(self):
        """Slide 5: while True → fill → trainNetworks → reset → bookkeeping → break."""
        results      = []
        totalEps     = 0
        fill_count   = 0

        # Build worker envs
        envs = [self.eval_env_fn(seed=self.seed + i)
                for i in range(self.numWorkers)]

        while True:
            self.rBuffer.fill(envs, self.world_model, self.pNetwork, self.vNetwork)
            self.trainNetworks()

            # Bookkeeping
            valid = np.where(self.rBuffer.episodeSteps > 0)[0]
            for ep in valid:
                self.trainEpisodeRewards.append(
                    float(self.rBuffer.episodeRewards[ep]))
            totalEps   += len(valid)
            fill_count += 1

            self.rBuffer.reset()
            self.performBookKeeping(train=True)

            # Stopping conditions (slide 5)
            timeOut          = (time.perf_counter() - self._t0) > 3600 * 4
            reachedMaxEp     = totalEps >= self.MAX_TRAIN_EPISODES
            reachedGoalReward = (
                len(self.trainEpisodeRewards) >= 10 and
                np.mean(self.trainEpisodeRewards[-10:]) > 1800
            )

            if timeOut or reachedGoalReward or reachedMaxEp:
                break

            results.append((fill_count, totalEps, float(
                np.mean(self.trainEpisodeRewards[-self.maxEpisodes:]))))

        for env in envs:
            try:
                env.close()
            except Exception:
                pass

        return results

    # ── trainNetworks ─────────────────────────────────────────────────────────
    def trainNetworks(self):
        zs, actions, rs, gaes, logps = self.rBuffer.returnElements()

        zs_t      = torch.tensor(zs,      dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        returns_t = torch.tensor(rs,      dtype=torch.float32)
        gaes_t    = torch.tensor(gaes,    dtype=torch.float32)
        old_logps = torch.tensor(logps,   dtype=torch.float32)

        # 🔥 IMPORTANT: get OLD values (detached)
        with torch.no_grad():
            old_values = self.vNetwork(zs_t).squeeze(-1)

        # ✅ Advantage normalization (KEEP THIS)
        gaes_t = (gaes_t - gaes_t.mean()) / (gaes_t.std() + 1e-8)

        nSamples = len(zs_t)
        batchSize_policy = int(self.policySampleRatio * nSamples)
        batchSize_value  = int(self.valueSampleRatio  * nSamples)

        # ───────────────── POLICY UPDATE ─────────────────
        for _ in range(self.policyOptimizationEpochs):

            idx = torch.randperm(nSamples)[:batchSize_policy]

            z_b   = zs_t[idx]
            a_b   = actions_t[idx]
            adv_b = gaes_t[idx]
            logp_old_b = old_logps[idx]

            _, logp, entropy, _ = self.pNetwork(z_b, a_b)

            # ratio
            ratio = torch.exp(logp - logp_old_b)

            # clipped surrogate
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio,
                                1.0 - self.policyClipRange,
                                1.0 + self.policyClipRange) * adv_b

            policy_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = -self.beta * entropy.mean()

            loss = policy_loss + entropy_loss

            self.policyOptimizerFn.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.pNetwork.parameters(), MAX_POLICY_GRAD)
            self.policyOptimizerFn.step()

            self.policyLosses.append(policy_loss.item())
            self.entropies.append(entropy.mean().item())

            # 🔥 CORRECT KL (important fix)
            with torch.no_grad():
                _, new_logp_all, _, _ = self.pNetwork(zs_t, actions_t)
                kl = (old_logps - new_logp_all).mean()

            if kl > self.policyStoppingKL:
                break

        # ───────────────── VALUE UPDATE ─────────────────
        for _ in range(self.valueOptimizationEpochs):

            idx = torch.randperm(nSamples)[:batchSize_value]

            z_b = zs_t[idx]
            r_b = returns_t[idx]
            v_old_b = old_values[idx]

            v_pred = self.vNetwork(z_b).squeeze(-1)

            # 🔥 CRITICAL FIX: proper clipped value loss
            v_clipped = v_old_b + torch.clamp(
                v_pred - v_old_b,
                -self.valueClipRange,
                self.valueClipRange
            )

            loss_unclipped = (v_pred - r_b) ** 2
            loss_clipped   = (v_clipped - r_b) ** 2

            value_loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

            self.valueOptimizerFn.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.vNetwork.parameters(), MAX_VALUE_GRAD)
            self.valueOptimizerFn.step()

            self.valueLosses.append(value_loss.item())

            # 🔥 FIXED stopping condition (was wrong before)
            with torch.no_grad():
                v_all = self.vNetwork(zs_t).squeeze(-1)
                mse = ((v_all - returns_t) ** 2).mean()

            if mse < self.valueStoppingMSE:
                break

    # ── evaluateAgent ─────────────────────────────────────────────────────────
    def evaluateAgent(self):
        """Greedy rollouts using world model for belief states."""
        rewards = []
        self.pNetwork.eval()
        self.world_model.eval()

        with torch.no_grad():
            for e in range(self.MAX_EVAL_EPISODE):
                rs   = 0.0
                env  = self.eval_env_fn(seed=self.seed + 90_000 + e)
                obs  = np.asarray(env.reset(), dtype=np.float32)
                wm_hx = self.world_model.init_hx(1)
                prev_action = 2  # FW

                done = False
                for c in count():
                    # Get belief state
                    z_t, wm_hx = self.world_model.encode_step(obs, prev_action, wm_hx)

                    z_t = torch.tensor(z_t, dtype=torch.float32).unsqueeze(0)
                    _, _, _, a_greedy = self.pNetwork.forward(z_t)
                    a = int(a_greedy.item())

                    obs, r, done = env.step(ACTIONS[a], render=self.eval_render)
                    obs  = np.asarray(obs, dtype=np.float32)
                    rs += float(r)
                    prev_action = a

                    if done:
                        rewards.append(rs)
                        break

                    if c >= self.maxEpisodeSteps - 1:
                        rewards.append(rs)
                        break

        self.performBookKeeping(train=False)
        self.pNetwork.train()
        self.world_model.train()
        return float(np.mean(rewards)), float(np.std(rewards))

    # ── performBookKeeping ────────────────────────────────────────────────────
    def performBookKeeping(self, train: bool = True):
        now = time.perf_counter() - self._t0
        self.wallTimes.append(now)

        if train and self.trainEpisodeRewards:
            recent = self.trainEpisodeRewards[-self.maxEpisodes:]
            r_arr  = np.array(recent)
            SEP    = "─" * 72
            print(f"\n{SEP}")
            print(f"  ▶  Total eps={len(self.trainEpisodeRewards)}"
                  f" / {self.MAX_TRAIN_EPISODES} ({now:.0f}s elapsed)")
            print(f"  RETURNS      "
                  f"mean={r_arr.mean():+9.2f}  std={r_arr.std():7.2f}  "
                  f"min={r_arr.min():+9.2f}  max={r_arr.max():+9.2f}")
            if self.policyLosses:
                pl = np.array(self.policyLosses[-20:])
                vl = np.array(self.valueLosses[-20:]) if self.valueLosses else np.array([0.0])
                ent= np.array(self.entropies[-20:])
                print(f"  POLICY LOSS  mean={pl.mean():+9.4f}")
                print(f"  VALUE  LOSS  mean={vl.mean():+9.4f}")
                print(f"  ENTROPY      mean={ent.mean():.4f}  "
                      f"({'exploring' if ent.mean() > 1.0 else 'converging'})")
            print(SEP, flush=True)
        elif not train and self.evalEpisodeRewards:
            print(f"  [eval]  mean={np.mean(self.evalEpisodeRewards[-self.MAX_EVAL_EPISODE:]):+.2f}",
                  flush=True)

    def save(self, path):
        torch.save(self.pNetwork.state_dict(), path)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",                type=str,   required=True)
    ap.add_argument("--out",                      type=str,   default="weights_rssm_ppo.pth")
    ap.add_argument("--wm_episodes",              type=int,   default=200)
    ap.add_argument("--wm_epochs",                type=int,   default=50)
    ap.add_argument("--wm_seq_len",               type=int,   default=20)
    ap.add_argument("--wm_hidden",                type=int,   default=64)
    ap.add_argument("--wm_z_dim",                 type=int,   default=32)
    ap.add_argument("--wm_lr",                    type=float, default=1e-3)
    ap.add_argument("--ppo_episodes",             type=int,   default=1000)
    ap.add_argument("--episodes_per_fill",        type=int,   default=16)
    ap.add_argument("--max_steps",                type=int,   default=200)
    ap.add_argument("--num_workers",              type=int,   default=4)
    ap.add_argument("--eval_episodes",            type=int,   default=5)
    ap.add_argument("--difficulty",               type=int,   default=0)
    ap.add_argument("--wall_obstacles",           action="store_true")
    ap.add_argument("--box_speed",                type=int,   default=2)
    ap.add_argument("--scaling_factor",           type=int,   default=5)
    ap.add_argument("--arena_size",               type=int,   default=500)
    ap.add_argument("--gamma",                    type=float, default=0.99)
    ap.add_argument("--lam",                      type=float, default=0.95)
    ap.add_argument("--beta",                     type=float, default=0.02)
    ap.add_argument("--ppo_lr_policy",            type=float, default=3e-4)
    ap.add_argument("--ppo_lr_value",             type=float, default=1e-3)
    ap.add_argument("--policy_epochs",            type=int,   default=8)
    ap.add_argument("--value_epochs",             type=int,   default=8)
    ap.add_argument("--policy_clip",              type=float, default=0.2)
    ap.add_argument("--value_clip",               type=float, default=0.2)
    ap.add_argument("--policy_sample_ratio",      type=float, default=0.8)
    ap.add_argument("--value_sample_ratio",       type=float, default=0.8)
    ap.add_argument("--policy_stopping_kl",       type=float, default=0.03)
    ap.add_argument("--value_stopping_mse",       type=float, default=1e6)
    ap.add_argument("--hDims_policy",             type=int,   nargs="+", default=[64, 64])
    ap.add_argument("--hDims_value",              type=int,   nargs="+", default=[64, 64])
    ap.add_argument("--seed",                     type=int,   default=42)
    ap.add_argument("--eval_render",              action="store_true")
    ap.add_argument("--wm_path",                  type=str,   default="world_model.pth")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    def env_fn(seed: int):
        return create_env(OBELIX, args, seed)

    # Phase 1: Train world model or load existing weights
    wm = WorldModel(obs_dim=N_OBS, action_dim=N_ACTIONS,
                    hidden_dim=args.wm_hidden, z_dim=args.wm_z_dim)
    if os.path.exists(args.wm_path):
        print(f"Loading existing world model from {args.wm_path}...")
        wm.load_state_dict(torch.load(args.wm_path, map_location="cpu"))
    else:
        print("Phase 1: Training World Model...")
        trainer = WorldModelTrainer(wm, env_fn, args)
        trainer.collect_data(args.wm_episodes)
        wm = trainer.train(args.wm_epochs, seq_len=args.wm_seq_len, lr=args.wm_lr)
        trainer.evaluate()
        torch.save(wm.state_dict(), args.wm_path)
        print(f"World model saved to {args.wm_path}")

    # Phase 2: Freeze world model, train PPO on z_t
    print("\nPhase 2: Training PPO with belief states...")
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    # Single env for eval (no multiprocessing needed)
    eval_env = env_fn(seed=args.seed)

    agent = PPO(
        env                      = eval_env,
        world_model              = wm,
        gamma                    = args.gamma,
        lam                      = args.lam,
        beta                     = args.beta,
        numWorkers               = args.num_workers,
        maxEpisodes              = args.episodes_per_fill,
        maxEpisodeSteps          = args.max_steps,
        updateFrequency          = 1,
        policyOptimizerFn        = optim.Adam,
        policyOptimizerLR        = args.ppo_lr_policy,
        policyOptimizationEpochs = args.policy_epochs,
        policyClipRange          = args.policy_clip,
        policySampleRatio        = args.policy_sample_ratio,
        policyStoppingKL         = args.policy_stopping_kl,
        valueOptimizerFn         = optim.Adam,
        valueOptimizerLR         = args.ppo_lr_value,
        valueOptimizationEpochs  = args.value_epochs,
        valueClipRange           = args.value_clip,
        valueSampleRatio         = args.value_sample_ratio,
        valueStoppingMSE         = args.value_stopping_mse,
        MAX_EVAL_EPISODE         = args.eval_episodes,
        z_dim                    = args.wm_z_dim,
        n_actions                = N_ACTIONS,
        hDims_policy             = tuple(args.hDims_policy),
        hDims_value              = tuple(args.hDims_value),
        MAX_TRAIN_EPISODES       = args.ppo_episodes,
        seed                     = args.seed,
        eval_render              = args.eval_render,
        eval_env_fn              = env_fn,
    )

    resultTrain, final_eval, training_time, wallclock = agent.runPPO()

    agent.save(args.out)
    print(f"\n[train_rssm_ppo] Saved → {args.out}")
    print(f"[train_rssm_ppo] Final eval score : {final_eval:+.2f}")
    print(f"[train_rssm_ppo] Training time    : {training_time:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()