from environments.rocketlander import RocketLander
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
    epsilon = 0.9
    total_reward = 0
    episode_number = 100
    lr = 0.1
    discount = 0.9

    for episode in range(episode_number):
        epsilon -= 0.1
        while 1:
            action = agent.get_actions(s, epsilon)

            s, r, done, info = env.step(action)
            total_reward += r

            agent.update_tables(s, r, lr, discount)

            if episode % 10 == 0 or (episode > 50 and episode % 5 == 0) or (episode > 100 and episode % 2 == 0) or episode > 150:
                env.render('human')
                env.refresh(render=False)

            if done:
                print('Episode:\t{}\tTotal Reward:\t{}'.format(episode, total_reward))
                total_reward = 0
                env.reset()
                break
