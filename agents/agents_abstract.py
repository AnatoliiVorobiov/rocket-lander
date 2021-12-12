import abc
from agents.pid_agent import PIDTuned
from agents.q_agent import FunctionApproximation


class AbstractAgent:
    @abc.abstractmethod
    def act(self, env, s):
        NotImplementedError()


class PIDAgent(AbstractAgent):
    def __init__(self):
        self.controller = PIDTuned()

    def act(self, env, s):
        return self.controller.pid_algorithm(s, y_target=None)


class QAgent(AbstractAgent):
    def __init__(self, state, low_discretization, load=None, epsilon=0.1, alpha=0.001):
        self.controller = FunctionApproximation(state, low_discretization, load, epsilon, alpha)

    def act(self, env, s):
        self.controller.update_state(s)
        return self.controller.act()
