import argparse
import cv2

import numpy as np

from obelix import OBELIX


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
    parser.add_argument(
        "--difficulty",
        help="difficulty level: 0=static, 2=blinking box, 3=moving+blinking",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--box_speed",
        help="speed of moving box (pixels/step) for difficulty>=3",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    bot = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )
    move_choice = ["L45", "L22", "FW", "R22", "R45"]
    user_input_choice = [ord("q"), ord("a"), ord("w"), ord("d"), ord("e")]
    bot.render_frame()
    episode_reward = 0
    for step in range(1, 2000):
        # random_step = np.random.choice(user_input_choice, 1, p=[0.05, 0.1, 0.7, 0.1, 0.05])[0]
        # # random_step = np.random.choice(user_input_choice, 1, p=[0.2, 0.2, 0.2, 0.2, 0.2])[0]
        # x = random_step
        x = cv2.waitKey(0)
        if x in user_input_choice:
            x = move_choice[user_input_choice.index(x)]
            sensor_feedback, reward, done = bot.step(x)
            episode_reward += reward
            print(step, sensor_feedback, episode_reward)
            if done:
                print("Episode done. Total score:", episode_reward)
                break
    cv2.waitKey(0)
    exit()
