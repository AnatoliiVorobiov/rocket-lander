from environments.rocketlander import RocketLander
from agent.pid import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Settings dict holds all the settings for the rocket lander environment.
    settings = {'Side Engines': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random'}  # (6000, -10000)}

    env = RocketLander(settings)
    s = env.reset()
    agent = PIDTuned2()
    display_name = 'PID2'
    episode_number = 200

    # Statistics
    total_reward = 0
    average_total_reward = 0
    average_total_rewards = []
    total_successes = 0

    print('Sample observation_space:', env.observation_space.sample())
    print('observation_space:', '(low = ', env.observation_space.low, '; high =', env.observation_space.high, ')')

    print('Sample action_space:', env.action_space.sample())
    print('observation_space:', '(low = ', env.action_space.low, '; high =', env.action_space.high, ')')

    for episode in range(episode_number):
        while 1:
            action = agent.pid_algorithm(s)

            s, r, done, info = env.step(action)
            total_reward += r

            env.render('human')
            env.refresh(render=False)

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
