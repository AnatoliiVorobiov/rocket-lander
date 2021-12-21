from environments.rocketlander import RocketLander
from agent.pid import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Settings dict holds all the settings for the rocket lander environment.
    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random'}  # (6000, -10000)}

    env = RocketLander(settings)
    s = env.reset()
    agent = PIDTuned1()
    display_name = 'PID1'
    episode_number = 200

    # Statistics
    total_reward = 0
    average_total_reward = 0
    average_total_rewards = []
    total_successes = 0

    for episode in range(episode_number):
        while 1:
            action = agent.pid_algorithm(s)

            s, r, done, info = env.step(action)
            total_reward += r

            # env.render('human')
            # env.refresh(render=False)

            if done:
                print('Episode:\t{}\tTotal Reward:\t{}'.format(episode, total_reward))
                average_total_reward = average_total_reward + 1 / (episode + 1) * (
                        total_reward - average_total_reward)
                average_total_rewards.append(average_total_reward)
                total_successes += info['success']
                total_reward = 0
                env.reset()
                break

    print(f'{display_name} success rate {total_successes}/{episode_number} ({total_successes / episode_number * 100}%)')
    plt.plot(average_total_rewards)
    plt.title(display_name + ' average reward')
    plt.show()
