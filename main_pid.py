from environments.rocketlander import RocketLander
import numpy as np
from agent.pid import *


if __name__ == "__main__":
    # Settings dict holds all the settings for the rocket lander environment.
    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random'}  # (6000, -10000)}

    env = RocketLander(settings)
    s = env.reset()

    agent = PIDTuned()

    left_or_right_barge_movement = np.random.randint(0, 2)
    total_reward = 0
    episode_number = 100

    for episode in range(episode_number):
        while 1:
            action = agent.pid_algorithm(s)

            s, r, done, info = env.step(action)
            total_reward += r

            env.render('human')
            env.refresh(render=False)

            if done:
                print('Episode:\t{}\tTotal Reward:\t{}'.format(episode, total_reward))
                total_reward = 0
                env.reset()
                break
