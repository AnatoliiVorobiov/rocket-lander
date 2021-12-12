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
