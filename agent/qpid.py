import math
import random

import numpy as np


S_SIZE = 8  # number of variables in a state
N_CONTROLLERS = 3
N_TABLES = N_CONTROLLERS * 3

N_CHUNKS = [5, 5, 5, 5, 5, 5, 2, 2]
MIN = [-1, -1, 0, 0, -1, -1, 0, 0]
MAX = [1, 1, 5, 5, 1, 1, 1, 1]
CHUNK_MULTIPLIER = list(N_CHUNKS[i] / (MAX[i] - MIN[i]) for i in range(S_SIZE))

K = [0, 0.05, 0.1, 0.3, 0.5, 0.8, 1]


class QPIDAgent:
    def __init__(self, load_path=None):
        if load_path is None:
            self.tables = self.new_tables()
        else:
            self.tables = self.load_tables(load_path)

        self.pids = list(PIDController() for _ in range(N_CONTROLLERS))

        self.prev_s = ()
        self.prev_k_indices = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    # ==== Q tables ====
    @staticmethod
    def discretize(s):
        """Discretize state from environment to int values from 0 to N_CHUNKS (excluded)"""
        discrete_s = []
        for i in range(S_SIZE):
            if s[i] <= MIN[i]:
                discrete_s.append(0)
            elif s[i] >= MAX[i]:
                discrete_s.append(N_CHUNKS[i] - 1)
            else:
                discrete_s.append(math.floor((s[i] - MIN[i]) * CHUNK_MULTIPLIER[i]))
        return tuple(discrete_s)

    @staticmethod
    def new_tables():
        # (number of tables, number of chunks for s[0], ..., number of chunks for s[7], number of possible coefficients)
        tables = np.zeros([N_TABLES] + N_CHUNKS + [len(K)])
        return tables

    def save_tables(self, save_path):
        # TODO
        pass

    def load_tables(self, load_path):
        # TODO
        return self.new_tables()

    def get_coefficients(self, s, eps):
        """Get coefficients for PID controllers.
        From Q-learning point of view, it just returns actions (one per table)"""
        s = self.discretize(s)
        if eps > 0 and random.random() < eps:
            k_indices = np.random.randint(len(K), size=N_TABLES)
        else:
            sl = (slice(0, N_TABLES),) + s
            k_indices = self.tables[sl].argmax(axis=1)

        coefficients = []
        for i in k_indices:
            coefficients.append(K[i])

        self.prev_s = s
        self.prev_k_indices = k_indices

        return coefficients

    def update_tables(self, new_s, reward, lr, discount):
        """Compute new values for tables based on previous action"""
        if len(self.prev_s) == 0:
            print('Attempting to update tables without experience')
            return

        new_s = self.discretize(new_s)

        prev_sl = (slice(0, N_TABLES),) + self.prev_s
        sl = (slice(0, N_TABLES),) + new_s

        td = lr * (reward + discount * self.tables[sl].max(axis=0) - self.tables[prev_sl][self.prev_k_indices])
        self.tables[prev_sl][self.prev_k_indices] = self.tables[prev_sl][self.prev_k_indices] + td

    # ==== PID + Q tables ====
    def get_actions(self, s, eps):
        """Compute action based on PID coefficients from Q-tables"""
        kp1, ki1, kd1, kp2, ki2, kd2, kp3, ki3, kd3 = self.get_coefficients(s, eps)
        dx, dy, vel_x, vel_y, theta, theta_dot, leg_contact_left, leg_contact_right = s

        xy_error = math.sqrt(dx*dx+dy*dy)
        eng = self.pids[0].compute_output(xy_error, kp1, ki1, kd1)
        nuzzle_angle = self.pids[1].compute_output(xy_error, kp2, ki2, kd2)

        theta_error = abs(theta) + abs(theta_dot)
        nitro = self.pids[2].compute_output(theta_error, kp3, ki3, kd3)

        # TODO check if values are in the action range
        return eng, nuzzle_angle, nitro


class PIDController:
    def __init__(self):
        self.accumulated_error = 0
        self.prev_error = 0

    def increment_integral_error(self, error, limit=3):
        self.accumulated_error = self.accumulated_error + error
        if self.accumulated_error > limit:
            self.accumulated_error = limit
        elif self.accumulated_error < limit:
            self.accumulated_error = -limit

    def compute_output(self, error, kp, ki, kd):
        self.increment_integral_error(error)
        dt_error = error - self.prev_error
        self.prev_error = error
        return kp * error + ki * self.accumulated_error + kd * dt_error
