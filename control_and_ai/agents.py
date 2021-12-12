from constants import *
import numpy as np
import abc
from environments.rocketlander import compute_derivatives


class EnvController:

    def draw(self, env):
        return None

    @abc.abstractclassmethod
    def act(self, env, s):
        NotImplementedError()


class PID_Controller(EnvController):
    def __init__(self):
        from .pid import PID_Benchmark
        self.controller = PID_Benchmark()

    def act(self, env, s):
        return self.controller.pid_algorithm(s, y_target=None)


class Q_Learning_Controller(EnvController):
    def __init__(self, state, low_discretization, load=None, epsilon=0.1, alpha=0.001):
        from .function_approximation_q_learning import FunctionApproximation
        self.controller = FunctionApproximation(state, low_discretization, load, epsilon, alpha)

    def act(self, env, s):
        self.controller.update_state(s)
        return self.controller.act()


class Q_Learning_Controller_Longer_State(EnvController):
    def __init__(self, state, low_discretization, load=None, epsilon=0.1, alpha=0.001):
        from .function_approximation_q_learning import FunctionApproximation
        self.controller = FunctionApproximation(state, low_discretization, load, epsilon, alpha)

    def act(self, env, s):
        state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
        self.controller.update_state(state)
        return self.controller.act()


class MPC_Controller(EnvController):
    def __init__(self, env, max_altitude=32, exponential_constant=0.05, ground_altitude=6, timespan=30,
                                timestep=1 / 60):
        from .mpc import MPC

        k, p = 100, 10

        Q_weights = np.array([0.25, 4, 0.25, 0.25, 365, 131])
        Q = np.eye(6) * Q_weights * k

        R_weights = np.array([0.1, 0.1, 10])
        R = np.eye(3) * R_weights * p

        mpc_settings = {'time_horizon': 30,
                        'control_horizon': 5,
                        'time_step': 1 / 30,
                        'time_span': 30,
                        'flight_time_span': 50,
                        'ground_altitude': 5.2,
                        'max_altitude': 26,
                        'x_target': 16.5,
                        'finite_differences_step_size': 50,
                        'Optimisation Type': 1,
                        'Q': Q,
                        'R': R}

        self.mpc_settings = mpc_settings
        self.controller = MPC(env)
        self.current_state = [0]*6
        self.current_action = [0]*3
        self.previous_state = [0]*6
        self.targets = [0]*6
        x_time, self.y_target_profile = self.controller.create_altitude_profile(max_altitude, exponential_constant,
                                                                                ground_altitude, timespan, timestep)

    def draw(self, env):
        env.draw_line(x=self.targets[XX], y=self.targets[YY], color=(0, 0, 0))
        env.draw_line(x=np.array(self.planned_states[0, :].value).squeeze(),
                      y=np.array(self.planned_states[1, :].value).squeeze(),
                      color=(1, 0, 0))

    def get_targets(self):

        targets = self.controller.guidance_target(state=self.current_state, final_x=self.mpc_settings['x_target'],
                                                  y_profile=self.y_target_profile,
                                                  time_horizon=self.mpc_settings['time_horizon'] + 1,
                                                  time_step=self.mpc_settings['time_step'])
        return targets

    def compute_derivatives_and_optimize(self, env, targets):
        A, B, x_0 = compute_derivatives(self.current_state, self.current_action,
                                        self.mpc_settings['finite_differences_step_size'])

        optimization_type = self.mpc_settings['Optimisation Type']
        ss = self.current_state
        prev_state = self.previous_state
        x = [0]*6
        u = [0]*3
        if optimization_type == 1:
            x, u = self.controller.optimize_linearized_model(A, B, ss, self.mpc_settings['Q'], self.mpc_settings['R'],
                                                             np.array(targets),
                                                             time_horizon=self.mpc_settings['time_horizon'],
                                                             verbose=False)
        elif optimization_type == 2:
            x, u = self.controller.optimize_analytical_model(ss, prev_state, self.mpc_settings['Q'],
                                                             self.mpc_settings['R'], np.array(targets),
                                                             time_horizon=self.mpc_settings['time_horizon'], verbose=False, env=env)
        elif optimization_type == 3:
            A = A[:-2, :-2]
            B = B[:-2, :-1]
            # A, B = compute_matrices_A_B_linearised_PID(env.untransformed_state, env.lander.mass, env.nozzle.angle)
            x, u = self.controller.optimize_with_PID(A, B, ss[:-2], self.mpc_settings['Q'],
                                                             self.mpc_settings['R'], np.array(targets)[:-2, :],
                                                time_horizon=self.mpc_settings['time_horizon'], verbose=False)
        return x, u

    def act(self, env, s):
        self.current_state = env.untransformed_state
        self.targets = self.get_targets()
        x, u = self.compute_derivatives_and_optimize(env, self.targets)
        self.planned_states = x
        self.previous_state = self.current_state
        a = [[0.5,0,0]] # Default Action
        if (u.value is not None):
            action = np.array(u.value).squeeze()

            if self.mpc_settings['Optimisation Type'] == 1 or self.mpc_settings['Optimisation Type'] == 2:
                a = action.T / [MAIN_ENGINE_POWER, SIDE_ENGINE_POWER, 1]
            else:
                a = action.T / [MAIN_ENGINE_POWER, SIDE_ENGINE_POWER]

        self.current_action = a[0]
        return a


class General_DDPG_Controller():
    def __init__(self,
                 env,
                 normal_state_FLAG,
                 untransformed_state_FLAG,
                 normalized_FLAG,
                 batch_size=100,
                 buffer_size=1000000,
                 actor_learning_rate=0.0001,
                 critic_learning_rate=0.001,
                 gamma=0.99,
                 retrain=False,
                 log_dir=None,
                 model_dir=None):

        from control_and_ai.DDPG.ddpg import DDPG
        from control_and_ai.DDPG.train import set_up
        from control_and_ai.DDPG.exploration import OUPolicy
        from constants import DEGTORAD
        from control_and_ai.DDPG.utils import Utils

        self.normalized_FLAG = normalized_FLAG
        self.normal_state_FLAG = normal_state_FLAG
        self.untransformed_state_FLAG = untransformed_state_FLAG

        FLAGS = set_up()

        action_bounds = [1, 1, 15 * DEGTORAD]

        eps = []
        eps.append(OUPolicy(0, 0.2, 0.4))
        eps.append(OUPolicy(0, 0.2, 0.4))
        eps.append(OUPolicy(0, 0.2, 0.4))

        FLAGS.retrain = False  # Restore weights if False
        FLAGS.test = True
        FLAGS.num_episodes = 300

        self.observation_space = env.observation_space.shape[0]
        self.util = Utils()
        if normalized_FLAG:
            self.sample_state_and_create_normalizer()

        self.controller = DDPG(action_bounds,
                               eps,
                               self.observation_space,
                               batch_size,
                               buffer_size,
                               actor_learning_rate,
                               critic_learning_rate,
                               gamma,
                               retrain,
                               log_dir,
                               model_dir)
        # test(env,controller,simulation_settings)

    def sample_state_and_create_normalizer(self):
        from environments.rocketlander import get_state_sample

        state_samples = get_state_sample(samples=2000, normal_state=self.normal_state_FLAG,
                                         untransformed_state=self.untransformed_state_FLAG)
        self.util.create_normalizer(state_sample=state_samples)


class Normalized_DDPG_Controller(General_DDPG_Controller, EnvController):
    """
    State = [x,z,x_dot,z_dot,theta,theta_dot] - Normalized
    """
    def __init__(self,
                 env,
                 batch_size=100,
                 buffer_size=1000000,
                 actor_learning_rate=0.0001,
                 critic_learning_rate=0.001,
                 gamma=0.99,
                 retrain=False,
                 log_dir=None,
                 model_dir=None):
        super(Normalized_DDPG_Controller, self)\
            .__init__(env,
                      True,
                      False,
                      True,
                      batch_size,
                      buffer_size,
                      actor_learning_rate,
                      critic_learning_rate,
                      gamma,
                      retrain,
                      log_dir,
                      model_dir)

    def act(self, env, s):
        state = self.util.normalize(s)
        action = self.controller.get_action(np.reshape(state, (1, self.observation_space)), explore=False)
        return action[0]


class Unnormalized_DDPG_Controller_Longer_State(General_DDPG_Controller, EnvController):
    """
    Unnormalized
    State = [x,z,x_dot,z_dot,theta,theta_dot,fuel,mass,barge left edge coordinates,barge right edge coordinates, landing coordinates]
    Used the following code when training:
        state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
        state = util.normalize(np.array(state))
    """

    def __init__(self,
                 env,
                 batch_size=100,
                 buffer_size=1000000,
                 actor_learning_rate=0.0001,
                 critic_learning_rate=0.001,
                 gamma=0.99,
                 retrain=False,
                 log_dir=None,
                 model_dir=None):
        super(Unnormalized_DDPG_Controller_Longer_State, self)\
            .__init__(env,
                      False,
                      False,
                      False,
                      batch_size,
                      buffer_size,
                      actor_learning_rate,
                      critic_learning_rate,
                      gamma,
                      retrain,
                      log_dir,
                      model_dir)

    def act(self, env, s):
        state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
        action = self.controller.get_action(np.reshape(state, (1, self.observation_space)), explore=False)
        return action[0]
