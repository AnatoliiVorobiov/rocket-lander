from environments.rocketlander import RocketLander
from constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT
import numpy as np
from agent.qpid import QPIDAgent


if __name__ == "__main__":
    # Settings dict holds all the settings for the rocket lander environment.
    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random'}  # (6000, -10000)}

    env = RocketLander(settings)
    s = env.reset()

    agent = QPIDAgent()

    left_or_right_barge_movement = np.random.randint(0, 2)
    epsilon = 0.1
    total_reward = 0
    episode_number = 5
    lr = 0.01
    discount = 0.9

    for episode in range(episode_number):
        while 1:
            action = agent.get_actions(s, epsilon)

            s, r, done, info = env.step(action)
            total_reward += r

            agent.update_tables(s, r, lr, discount)

            env.render('human')
            env.refresh(render=False)

            # When should the barge move? Water movement, dynamics etc can be simulated here.
            if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
                # Random Force on rocket to simulate wind.
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)

            # Touch down or pass abs(THETA_LIMIT)
            if done:
                print('Episode:\t{}\tTotal Reward:\t{}'.format(episode, total_reward))
                total_reward = 0
                env.reset()
                break
