"""PPO trainer for OBELIX with RewardShaper — matches lecture pseudocode exactly.

Implements all 8 slides:
  MultiEnv        — multiprocessing worker pool (pipes + mp.Process)
  EpisodeBuffer   — fill, returnElements, updateBufferElement, reset
  PolicyNetwork   — feedforward inputLayer + hLayers + out
  ValueNetwork    — feedforward inputLayer + hLayers + out
  PPO             — initBookKeeping, runPPO, trainAgent, trainNetworks,
                    evaluateAgent, performBookKeeping
  RewardShaper    — shapes rewards based on task sub-behaviors

Run:
  python train_ppo_rs.py --obelix_py ./obelix.py --out weights_ppo.pth
"""

from __future__ import annotations

import argparse
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

SONAR_ANGLES = np.deg2rad([-90, -90, 0, 0, 0, 0, 90, 90])

class RewardShaper:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        # Reset episode state for one-shot bonuses and counters
        self.search_first_sonar = False
        self.align_consecutive_turns = 0
        self.push_first_contact = False
        self.push_streak = 0
        self.unstuck_steps = 0

    def step(self,
             obs_prev: np.ndarray,   # obs BEFORE action (18-bit)
             obs_next: np.ndarray,   # obs AFTER action (18-bit, from env)
             action: int,
             env_reward: float,
             done: bool,
             fsm_was_active: bool    # True if a stuck-escape FSM chose this action
             ) -> float:             # shaped reward to use instead of env_reward

        shaped_reward = env_reward

        # Determine mode from obs_prev (except UNSTUCK which uses obs_next[17])
        if obs_next[17] == 1:
            mode = 'UNSTUCK'
        elif obs_prev[16] == 1 and obs_prev[17] == 0:
            mode = 'PUSH'
        elif np.any(obs_prev[0:16] == 1):
            mode = 'ALIGN'
        else:
            mode = 'SEARCH'

        # SEARCH mode: box not visible, encourage scanning
        if mode == 'SEARCH':
            if action in {0, 1, 3, 4}:  # turning actions
                shaped_reward += 0.10  # reward turning to scan
            if action == 2 and np.all(obs_prev[0:16] == 0):
                shaped_reward -= 0.40  # penalize blind forward charging
            if not self.search_first_sonar and np.any(obs_next[0:16] == 1):
                shaped_reward += 0.30  # bonus for first sonar detection
                self.search_first_sonar = True

        # ALIGN mode: sonar firing but no IR, encourage centering box
        elif mode == 'ALIGN':
            # Compute sonar centroid (circular mean of firing units)
            sin_sum = 0.0
            cos_sum = 0.0
            total_weight = 0
            for i in range(8):
                if obs_prev[2*i] == 1 or obs_prev[2*i+1] == 1:
                    angle = SONAR_ANGLES[i]
                    sin_sum += np.sin(angle)
                    cos_sum += np.cos(angle)
                    total_weight += 1
            if total_weight > 0:
                mean_angle = np.arctan2(sin_sum, cos_sum)
                theta_err = mean_angle  # error from dead ahead (0)
                alignment_reward = (1 - abs(theta_err) / np.pi) * 0.10
                shaped_reward += alignment_reward  # reward proportional to alignment
                if action == 2:  # forward
                    if abs(theta_err) < np.pi/4:
                        shaped_reward += 0.08  # reward aligned forward
                    else:
                        shaped_reward -= 0.04  # penalize misaligned forward
            # Track consecutive turns to prevent spinning
            if action in {0, 1, 3, 4}:
                self.align_consecutive_turns += 1
                if self.align_consecutive_turns > 3:
                    shaped_reward -= 0.20  # increased spin penalty
            else:
                self.align_consecutive_turns = 0

        # PUSH mode: IR firing and not stuck, encourage maintaining contact
        elif mode == 'PUSH':
            contact = obs_next[16] == 1 and obs_next[17] == 0
            if contact:
                if not self.push_first_contact:
                    shaped_reward += 0.50  # bonus for first confirmed contact
                    self.push_first_contact = True
                self.push_streak += 1
                shaped_reward += 0.06  # reward sustained contact
                shaped_reward += 0.02 * min(self.push_streak, 20)  # streak multiplier
            else:
                if self.push_first_contact:
                    shaped_reward -= 0.15  # penalize losing contact after touch
                self.push_streak = 0

        # UNSTUCK mode: stuck flag firing, encourage turning away
        elif mode == 'UNSTUCK':
            shaped_reward -= 1.00  # base stuck penalty
            self.unstuck_steps += 1
            if self.unstuck_steps > 2:
                shaped_reward -= 0.08 * (self.unstuck_steps - 2)  # escalating penalty
            if action in {0, 1, 3, 4} and not fsm_was_active:
                shaped_reward += 0.04  # reward turning when stuck
            if action == 2 and not fsm_was_active:
                shaped_reward -= 0.40  # penalize forward when stuck

        # Terminal rewards
        if done:
            if env_reward >= 1000:
                shaped_reward += 10.00  # success bonus
            else:
                shaped_reward -= 0.50  # timeout penalty

        return shaped_reward


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
#  MultiEnv  (slide 1, 2, 3)
# ─────────────────────────────────────────────────────────────────────────────
class MultiEnv:
    """Parallel environment pool using mp.Pipe + mp.Process.

    Each worker runs an OBELIX env in its own process.
    The main process communicates via pipes sending ('cmd', kwargs) tuples.
    """

    def __init__(self, envName, N_WORKERS: int, seed: int, args):
        self.N_WORKERS = N_WORKERS
        self.args      = args
        self.seed      = seed
        self.envName   = envName   # OBELIX class

        # Each pipe entry is (parent_end, child_end)
        self.pipes = [mp.Pipe() for _ in range(N_WORKERS)]

        self.workers = []
        for id_ in range(N_WORKERS):
            p = mp.Process(
                target=self.work,
                args=(id_, self.pipes[id_][1]),   # child end
                daemon=True,
            )
            self.workers.append(p)
        for w in self.workers:
            w.start()

    # ── send_msg / broadcast ──────────────────────────────────────────────────
    def send_msg(self, m, id_: int):
        parent_end, _ = self.pipes[id_]
        parent_end.send(m)

    def broadcast(self, m):
        for parent_end, _ in self.pipes:
            parent_end.send(m)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, id_=None, **kwargs):
        if id_ is not None:
            parent_end, _ = self.pipes[id_]
            self.send_msg(('reset', {}), id_)
            s = parent_end.recv()
            return s
        self.broadcast(('reset', kwargs))
        s_list = [parent_end.recv() for parent_end, _ in self.pipes]
        return s_list

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, actions):
        for id_ in range(self.N_WORKERS):
            self.send_msg(('step', {'action': actions[id_]}), id_)
        results = []
        for id_ in range(self.N_WORKERS):
            parent_end, _ = self.pipes[id_]
            s, r, done = parent_end.recv()
            results.append((s, r, done))
        return np.array(results, dtype=object)

    # ── close ─────────────────────────────────────────────────────────────────
    def close(self, **kwargs):
        self.broadcast(('close', {}))

    # ── timeOut ───────────────────────────────────────────────────────────────
    def timeOut(self, **kwargs):
        pass   # timeout logic handled in trainAgent

    # ── work  (runs in child process) ─────────────────────────────────────────
    def work(self, id_: int, worker_end):
        env = create_env(self.envName, self.args, seed=self.seed + id_)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                obs = env.reset()
                worker_end.send(np.asarray(obs, dtype=np.float32))
            elif cmd == 'step':
                action_str = kwargs['action']
                obs, r, done = env.step(action_str, render=False)
                worker_end.send((
                    np.asarray(obs, dtype=np.float32),
                    float(r),
                    bool(done),
                ))
            else:   # 'close'
                del env
                worker_end.close()
                break


# ─────────────────────────────────────────────────────────────────────────────
#  Networks  (slides 6 & 7)
# ─────────────────────────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    """inputLayer → hLayers → out, activation = F.relu."""

    def __init__(self, stateDim=N_OBS, n_actions=N_ACTIONS,
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

    def __init__(self, stateDim=N_OBS, hDims=(64, 64), activationFn=F.relu):
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
#  EpisodeBuffer  (slides from previous set)
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeBuffer:
    def __init__(self, gamma: float, lam: float,
                 stateDim: int, numWorkers: int,
                 maxEpisodes: int, maxEpisodeSteps: int):
        self.gamma           = gamma
        self.lam             = lam
        self.stateDim        = stateDim
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

    def fill(self, envs, pNetwork: PolicyNetwork, vNetwork: ValueNetwork, shapers):
        """Collect episodes from worker envs until buffer is full."""
        numWorkers = self.numWorkers
        T          = self.maxEpisodeSteps

        ss = np.stack([np.asarray(env.reset(), dtype=np.float32)
                       for env in envs])
        for shaper in shapers:
            shaper.reset()

        bufferFull = False
        wRewards   = np.zeros((numWorkers, T), dtype=np.float32)
        wSteps     = np.zeros(numWorkers, dtype=np.int64)

        pNetwork.eval(); vNetwork.eval()

        while not bufferFull:
            with torch.no_grad():
                ss_t = torch.tensor(ss, dtype=torch.float32)
                actions_t, logp_t, _, _ = pNetwork.forward(ss_t)
                vs_t = vNetwork.forward(ss_t).squeeze(-1)

            actions = actions_t.numpy()
            logp_as = logp_t.numpy()

            for id_ in range(numWorkers):
                epID = self.currentEpisodeIDs[id_]
                t    = wSteps[id_]
                self.bufferStates [epID, t] = ss[id_]
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
                obs_prev = ss[id_]
                obs_next = sNexts[id_]
                action = actions[id_]
                env_r = rs[id_]
                done_flag = dones[id_] == 1
                shaped_r = shapers[id_].step(obs_prev, obs_next, action, env_r, done_flag, False)
                wRewards[id_, wSteps[id_]] = shaped_r

            for id_ in range(numWorkers):
                if wSteps[id_] + 1 == T:
                    dones[id_] = 1

            if dones.sum():
                dones_ids = np.where(dones)[0]
                nValues   = np.zeros(numWorkers, dtype=np.float32)
                with torch.no_grad():
                    sn_t = torch.tensor(sNexts[dones_ids], dtype=torch.float32)
                    nValues[dones_ids] = vNetwork.forward(sn_t).squeeze(-1).numpy()

                for id_ in dones_ids:
                    sNexts[id_] = np.asarray(envs[id_].reset(), dtype=np.float32)
                    shapers[id_].reset()

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
                    epStates = self.bufferStates[epID, :ep_T]
                    with torch.no_grad():
                        epV = vNetwork.forward(
                            torch.tensor(epStates, dtype=torch.float32)
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

            ss      = sNexts
            wSteps += 1

        pNetwork.train(); vNetwork.train()


# ─────────────────────────────────────────────────────────────────────────────
#  PPO class  (slides 4–8)
# ─────────────────────────────────────────────────────────────────────────────
class PPO:
    def __init__(
        self,
        env,                        # single OBELIX env for eval
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
        stateDim:             int   = N_OBS,
        n_actions:            int   = N_ACTIONS,
        hDims_policy:         tuple = (64, 64),
        hDims_value:          tuple = (64, 64),
        # training budget
        MAX_TRAIN_EPISODES:   int   = 2000,
        seed:                 int   = 3,
        eval_render:          bool  = False,
        eval_env_fn                 = None,
        curriculum_interval:  int   = 10,
        args                        = None,
    ):
        self.env                       = env
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
        self.curriculum_interval       = curriculum_interval
        self.args                      = args

        self.pNetwork = PolicyNetwork(stateDim, n_actions, hDims_policy)
        self.vNetwork = ValueNetwork(stateDim, hDims_value)

        self.policyOptimizerFn = policyOptimizerFn(
            self.pNetwork.parameters(), lr=policyOptimizerLR)
        self.valueOptimizerFn  = valueOptimizerFn(
            self.vNetwork.parameters(), lr=valueOptimizerLR)

        self.rBuffer = EpisodeBuffer(
            gamma=gamma, lam=lam,
            stateDim=stateDim, numWorkers=numWorkers,
            maxEpisodes=maxEpisodes, maxEpisodeSteps=maxEpisodeSteps,
        )

        self.shapers = [RewardShaper() for _ in range(self.numWorkers)]

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
        use_walls    = False  # start without walls

        # Build worker envs
        envs = [self.eval_env_fn(seed=self.seed + i)
                for i in range(self.numWorkers)]

        while True:
            self.rBuffer.fill(envs, self.pNetwork, self.vNetwork, self.shapers)
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

            # Curriculum: toggle walls every curriculum_interval episodes
            new_use_walls = (totalEps // self.curriculum_interval) % 2 == 1
            if new_use_walls != use_walls:
                use_walls = new_use_walls
                self.args.wall_obstacles = use_walls
                # Recreate envs with new wall setting
                envs = [self.eval_env_fn(seed=self.seed + i)
                        for i in range(self.numWorkers)]
                print(f"[Curriculum] Walls {'ON' if use_walls else 'OFF'} at totalEps={totalEps}")

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
        ss, actions, rs, gaes, logps = self.rBuffer.returnElements()

        ss_t      = torch.tensor(ss,      dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        returns_t = torch.tensor(rs,      dtype=torch.float32)
        gaes_t    = torch.tensor(gaes,    dtype=torch.float32)
        old_logps = torch.tensor(logps,   dtype=torch.float32)

        # 🔥 IMPORTANT: get OLD values (detached)
        with torch.no_grad():
            old_values = self.vNetwork(ss_t).squeeze(-1)

        # ✅ Advantage normalization (KEEP THIS)
        gaes_t = (gaes_t - gaes_t.mean()) / (gaes_t.std() + 1e-8)

        nSamples = len(ss_t)
        batchSize_policy = int(self.policySampleRatio * nSamples)
        batchSize_value  = int(self.valueSampleRatio  * nSamples)

        # ───────────────── POLICY UPDATE ─────────────────
        for _ in range(self.policyOptimizationEpochs):

            idx = torch.randperm(nSamples)[:batchSize_policy]

            s_b   = ss_t[idx]
            a_b   = actions_t[idx]
            adv_b = gaes_t[idx]
            logp_old_b = old_logps[idx]

            _, logp, entropy, _ = self.pNetwork(s_b, a_b)

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
                _, new_logp_all, _, _ = self.pNetwork(ss_t, actions_t)
                kl = (old_logps - new_logp_all).mean()

            if kl > self.policyStoppingKL:
                break

        # ───────────────── VALUE UPDATE ─────────────────
        for _ in range(self.valueOptimizationEpochs):

            idx = torch.randperm(nSamples)[:batchSize_value]

            s_b = ss_t[idx]
            r_b = returns_t[idx]
            v_old_b = old_values[idx]

            v_pred = self.vNetwork(s_b).squeeze(-1)

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
                v_all = self.vNetwork(ss_t).squeeze(-1)
                mse = ((v_all - returns_t) ** 2).mean()

            if mse < self.valueStoppingMSE:
                break


    # ── evaluateAgent ─────────────────────────────────────────────────────────
    def evaluateAgent(self):
        """Slide 8: greedy rollouts, return (mean, std)."""
        rewards = []
        self.pNetwork.eval()

        with torch.no_grad():
            for e in range(self.MAX_EVAL_EPISODE):
                rs   = 0.0
                env  = self.eval_env_fn(seed=self.seed + 90_000 + e)
                s    = np.asarray(env.reset(), dtype=np.float32)
                done = False

                for c in count():
                    s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                    _, _, _, a_greedy = self.pNetwork.forward(s_t)
                    a = int(a_greedy.item())

                    s, r, done = env.step(ACTIONS[a], render=True)  # always render eval
                    s  = np.asarray(s, dtype=np.float32)
                    rs += float(r)

                    if done:
                        rewards.append(rs)
                        break

                    if c >= self.maxEpisodeSteps - 1:
                        rewards.append(rs)
                        break

        self.performBookKeeping(train=False)
        self.pNetwork.train()
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
                  f" / {self.MAX_TRAIN_EPISODES}"
                  f"   ({now:.0f}s elapsed)")
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


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",                type=str,   required=True)
    ap.add_argument("--out",                      type=str,   default="weights_ppo_rs.pth")
    ap.add_argument("--total_episodes",           type=int,   default=2000)
    ap.add_argument("--episodes_per_fill",        type=int,   default=16)
    ap.add_argument("--max_steps",                type=int,   default=200)
    ap.add_argument("--num_workers",              type=int,   default=4)
    ap.add_argument("--eval_episodes",            type=int,   default=5)
    ap.add_argument("--difficulty",               type=int,   default=0)
    ap.add_argument("--wall_obstacles",           action="store_true")
    ap.add_argument("--box_speed",                type=int,   default=2)
    ap.add_argument("--scaling_factor",           type=int,   default=5)
    ap.add_argument("--arena_size",               type=int,   default=500)
    ap.add_argument("--gamma",                    type=float, default=0.98)
    ap.add_argument("--lam",                      type=float, default=0.95)
    ap.add_argument("--beta",                     type=float, default=0.001)
    ap.add_argument("--lr_policy",                type=float, default=3e-4)
    ap.add_argument("--lr_value",                 type=float, default=1e-3)
    ap.add_argument("--policy_epochs",            type=int,   default=10)
    ap.add_argument("--value_epochs",             type=int,   default=10)
    ap.add_argument("--policy_clip",              type=float, default=0.2)
    ap.add_argument("--value_clip",               type=float, default=0.2)
    ap.add_argument("--policy_sample_ratio",      type=float, default=0.8)
    ap.add_argument("--value_sample_ratio",       type=float, default=0.8)
    ap.add_argument("--policy_stopping_kl",       type=float, default=0.03)
    ap.add_argument("--value_stopping_mse",       type=float, default=1e6)
    ap.add_argument("--hDims_policy",             type=int,   nargs="+", default=[64, 64])
    ap.add_argument("--hDims_value",              type=int,   nargs="+", default=[64, 64])
    ap.add_argument("--seed",                     type=int,   default=3)
    ap.add_argument("--eval_render",              action="store_true")
    ap.add_argument("--curriculum_interval",      type=int,   default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    def env_fn(seed: int):
        return create_env(OBELIX, args, seed)

    # Single env for eval (no multiprocessing needed)
    eval_env = env_fn(seed=args.seed)

    agent = PPO(
        env                      = eval_env,
        gamma                    = args.gamma,
        lam                      = args.lam,
        beta                     = args.beta,
        numWorkers               = args.num_workers,
        maxEpisodes              = args.episodes_per_fill,
        maxEpisodeSteps          = args.max_steps,
        updateFrequency          = 1,
        policyOptimizerFn        = optim.Adam,
        policyOptimizerLR        = args.lr_policy,
        policyOptimizationEpochs = args.policy_epochs,
        policyClipRange          = args.policy_clip,
        policySampleRatio        = args.policy_sample_ratio,
        policyStoppingKL         = args.policy_stopping_kl,
        valueOptimizerFn         = optim.Adam,
        valueOptimizerLR         = args.lr_value,
        valueOptimizationEpochs  = args.value_epochs,
        valueClipRange           = args.value_clip,
        valueSampleRatio         = args.value_sample_ratio,
        valueStoppingMSE         = args.value_stopping_mse,
        MAX_EVAL_EPISODE         = args.eval_episodes,
        stateDim                 = N_OBS,
        n_actions                = N_ACTIONS,
        hDims_policy             = tuple(args.hDims_policy),
        hDims_value              = tuple(args.hDims_value),
        MAX_TRAIN_EPISODES       = args.total_episodes,
        seed                     = args.seed,
        eval_render              = args.eval_render,
        eval_env_fn              = env_fn,
        curriculum_interval      = args.curriculum_interval,
        args                     = args,
    )

    resultTrain, final_eval, training_time, wallclock = agent.runPPO()

    torch.save(agent.pNetwork.cpu().state_dict(), args.out)
    print(f"\n[train_ppo_rs] Saved → {args.out}")
    print(f"[train_ppo_rs] Final eval score : {final_eval:+.2f}")
    print(f"[train_ppo_rs] Training time    : {training_time:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()