import _pickle
import logging
from q_agent import *
from environments.rocketlander import RocketLander


def rocket_rl_function_approximation(env, settings: dict, logger,
                                     load_path=None, save_path=None, low_discretization=True):
    if settings['Test']:
        print("Testing rocket_rl_function_approximation with load_path = {0}, save_path = {1}".format(load_path,
                                                                                                      save_path))
    else:
        print("Training rocket_rl_function_approximation with load_path = {0}, save_path = {1}".format(load_path,
                                                                                                       save_path))
    s = env.reset()
    reinforced_control = FunctionApproximation(s, load=load_path, low_discretization=low_discretization,
                                               epsilon=0.001, alpha=0.001)
    max_steps = 1000

    def train():
        for episode in range(settings['Episodes']):
            if episode % 50 == 0:
                print('Episode', episode)
            steps = 0
            while steps < max_steps:
                a = reinforced_control.act()
                s, r, done, info = env.step(a)
                s = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
                reinforced_control.learn(s, r)
                if episode % 50 == 0 or settings['Render']:
                    env.refresh(render=True)

                if done:
                    logger.info('Episode:\t{0}\tReward:\t{1}'.format(episode, reinforced_control.total_reward))

                    reinforced_control.reset()
                    break

                steps += 1

            if episode % 50 == 0 and save_path is not None:
                logger.info('Saving Model at Episode:\t{0}'.format(episode))
                with open(save_path, "wb") as f:
                    print(f'save shape: {reinforced_control.theta.shape}')
                    _pickle.dump(reinforced_control.theta, f)

    def test():
        episode = 0
        while episode < settings['Episodes']:
            a = reinforced_control.act()
            s, r, done, info = env.step(a)
            if settings['Render']:
                env.refresh(render=True)

            logger.info('Episode:\t{0}\tReward:\t{1}'.format(episode, reinforced_control.total_reward))

            if done:
                env.reset()
                reinforced_control.reset()
                episode += 1

    if isinstance(settings.get('Test'), bool) and settings['Test']:
        test()
    else:
        train()


def train_low_discretization_rl():
    print("Training LOW Discretization RL-Function Approximator")
    load_path = None
    save_path = './q_trained.p'
    rocket_rl_function_approximation(env[0], settings=simulation_settings, logger=logger, load_path=load_path,
                                     save_path=save_path, low_discretization=True)


def train_high_discretization_rl():
    print("Training HIGH Discretization RL-Function Approximator")
    load_path = None
    save_path = './q_hd_trained.p'
    rocket_rl_function_approximation(env[0], settings=simulation_settings, logger=logger, load_path=load_path,
                                     save_path=save_path, low_discretization=False)


if __name__ == '__main__':
    verbose = True
    logger = logging.getLogger(__name__)
    if verbose:
        logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)
        logging.basicConfig(format='%(asctime)s - %(message)s\t', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.ERROR)
        logger.setLevel(logging.INFO)

    simulation_settings = {'Side Engines': True,
                           'Clouds': True,
                           'Vectorized Nozzle': True,
                           'Graph': False,
                           'Render': False,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2,
                           # 'Initial Coordinates': (0.8,0.5,0),
                           'Test': False,
                           'Episodes': 51}

    evo_strategy_parameters = {
        'population_size': 100,
        'action_size': 3,
        'noise_standard_deviation': 0.1,
        'number_of_generations': 1000,
        'learning_rate': 0.00025,
        'state_size': 8,
        'max_num_actions': 250
    }

    env = []
    for i in range(evo_strategy_parameters['population_size'] + 1):
        env.append(RocketLander(simulation_settings))

    # train_low_discretization_rl()
    train_high_discretization_rl()
