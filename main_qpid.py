from environments.rocketlander import RocketLander
from agent.qpid import QPIDAgent
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
    agent = QPIDAgent()
    episode_number = 200

    # Q-tables parameters
    epsilon = 0.9
    lr = 0.2
    discount = 0.9

    # Statistics
    total_reward = 0
    average_total_reward = 0
    average_total_rewards = []
    total_successes = 0

    for episode in range(episode_number):
        epsilon -= 0.01
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
                average_total_reward = average_total_reward + 1 / (episode + 1) * (
                            total_reward - average_total_reward)
                average_total_rewards.append(average_total_reward)
                total_successes += info['success']
                total_reward = 0
                env.reset()
                break

        print(f'Success rate {total_successes}/{episode_number} ({total_successes / episode_number * 100}%)')
        plt.plot(average_total_rewards)
        plt.show()
