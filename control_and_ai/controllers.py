"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Rocket Lander Controllers that are called for evaluation.
"""

import abc

''' Main Environment Parent Class'''
class EnvController():

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
    def __init__(self, load_path, simulation_settings, low_discretization):
        from .function_approximation_q_learning import FunctionApproximation
        assert load_path is not None
        dummy_state = [0.5, 1, 0, 0, 0, 0, 0, 0]
        self.controller = FunctionApproximation(dummy_state,
                                                load=load_path,
                                                low_discretization=low_discretization,
                                                epsilon=0)

    def act(self, env, s):
        self.controller.update_state(s)
        return self.controller.act()
