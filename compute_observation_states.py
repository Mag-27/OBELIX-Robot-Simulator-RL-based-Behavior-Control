# this script iterates over the environment and plots the observation states on the grid

import argparse
import cv2

import numpy as np

from obelix import OBELIX

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sf",
        "--scaling_factor",
        help="decides the scaling of the bot and the environment",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--arena_size", help="arena side length in pixels", type=int, default=500
    )
    parser.add_argument(
        "--max_steps", help="maximum steps per episode", type=int, default=2000
    )
    parser.add_argument(
        "--wall_obstacles", help="add static wall obstacles", action="store_true"
    )
    args = parser.parse_args()

    bot = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
    )
    move_choice = ["L45", "L22", "FW", "R22", "R45"]
    user_input_choice = [ord("q"), ord("a"), ord("w"), ord("d"), ord("e")]
    episode_reward = 0

    set_of_observation_states = []
    count_of_observation_states = []
    for step in range(1, 2000):
        random_step = np.random.choice(
            user_input_choice, 1, p=[0.05, 0.1, 0.7, 0.1, 0.05]
        )[0]
        # # random_step = np.random.choice(user_input_choice, 1, p=[0.2, 0.2, 0.2, 0.2, 0.2])[0]
        x = random_step
        # x = cv2.waitKey(0)
        if x in user_input_choice:
            x = move_choice[user_input_choice.index(x)]
            sensor_feedback, reward, done = bot.step(x, render=False)
            episode_reward += reward
            print(step, sensor_feedback, episode_reward)
            if tuple(sensor_feedback.tolist()) not in set_of_observation_states:
                # breakpoint()
                set_of_observation_states.append(tuple(sensor_feedback.tolist()))
                count_of_observation_states.append(1)
            else:
                # increase the count of the observation state
                count_of_observation_states[
                    set_of_observation_states.index(tuple(sensor_feedback.tolist()))
                ] += 1

        if step % 100 == 0:
            print("Number of observation states: ", len(set_of_observation_states))
            print("Observation states: ", set_of_observation_states)
            print("Count of observation states: ", count_of_observation_states)
            # plot matshow of the observation states with the count of the observation states
            # create a matrix of zeros
            # set_of_observation_states = np.asarray(set_of_observation_states)
            # # scale the observations with the count of the observation states
            # set_of_observation_states = set_of_observation_states*np.repeat(count_of_observation_states, len(set_of_observation_states[0])).reshape(len(set_of_observation_states), len(set_of_observation_states[0]))
            plt.matshow(
                set_of_observation_states
                * np.repeat(
                    count_of_observation_states, len(set_of_observation_states[0])
                ).reshape(
                    len(set_of_observation_states), len(set_of_observation_states[0])
                )
            )
            plt.colorbar()
            plt.title("Observation states with count")
            plt.show()
            plt.close()
    exit()
