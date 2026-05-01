import importlib.util
import numpy as np
import argparse

from obelix import OBELIX


ACTIONS = ("L45", "L22", "FW", "R22", "R45")


def load_policy(agent_file):

    spec = importlib.util.spec_from_file_location("agent", agent_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod.policy


def evaluate(agent_policy, runs, seed, max_steps, wall_obstacles):

    rewards = []

    rng = np.random.default_rng(seed)

    for run in range(runs):

        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=3,
            box_speed=2,
            seed=seed + run + 10
        )

        obs = np.asarray(env.reset(), dtype=np.float32)

        total_reward = 0

        for step in range(max_steps):

            action = agent_policy(obs, rng)
        
            obs, r, done = env.step(action, render=True)

            obs = np.asarray(obs, dtype=np.float32)

            total_reward += r

            if done:
                break

        print(f"Run {run+1} reward: {total_reward:.2f}")

        rewards.append(total_reward)

    print("\nAverage reward:", np.mean(rewards))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--agent_file", type=str, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--wall_obstacles", action="store_true")

    args = parser.parse_args()

    policy = load_policy(args.agent_file)

    evaluate(
        policy,
        args.runs,
        args.seed,
        args.max_steps,
        args.wall_obstacles
    )


if __name__ == "__main__":
    main()