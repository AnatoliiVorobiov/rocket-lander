import math
import random

import numpy as np


S_SIZE = 8  # number of variables in a state
N_CONTROLLERS = 3
N_TABLES = N_CONTROLLERS * 3

N_CHUNKS = [ 4,  3,  4,  4,     4,  2, 2, 2]
MIN =      [-2, -1, -3, -1, -3.14, -1, 0, 0]
MAX =      [ 2,  2,  3,  1,  3.14,  1, 1, 1]
CHUNK_MULTIPLIER = list(N_CHUNKS[i] / (MAX[i] - MIN[i]) for i in range(S_SIZE))

N_K = 5
K = [list(i * 0.001 * 2 / N_K for i in range(N_K)),
     list(i * 0.001 * 2 / N_K for i in range(N_K)),
     list(i * 0.001 * 2 / N_K for i in range(N_K)),
     list(i * 0.08 * 2 / N_K for i in range(N_K)),
     list(i * 0.001 * 2 / N_K for i in range(N_K)),
     list(i * 10 * 2 / N_K for i in range(N_K)),
     list(i * 5 * 2 / N_K for i in range(N_K)),
     list(i * 5.5 * 2 / N_K for i in range(N_K)),
     list(i * 6 * 2 / N_K for i in range(N_K)),
     ]


class QPIDAgent:
    def __init__(self, load_path=None):
        if load_path is None:
            self.tables = self.new_tables()
        else:
            self.tables = self.load_tables(load_path)

        self.pids = list(PIDController() for _ in range(N_CONTROLLERS))

        self.prev_s_d = ()
        self.prev_k_indices = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    # ==== Q tables ====
    @staticmethod
    def discretize(s):
        """Discretize state from environment to int values from 0 to N_CHUNKS (excluded)"""
        discrete_s = []

        if s[0] < -0.3:
            discrete_s.append(0)
        elif s[0] < 0:
            discrete_s.append(1)
        elif s[0] < 0.3:
            discrete_s.append(2)
        else:
            discrete_s.append(3)

        if s[1] < 0:
            discrete_s.append(0)
        elif s[1] < 0.1:
            discrete_s.append(1)
        else:
            discrete_s.append(2)

        for i in range(2, S_SIZE):
            if s[i] <= MIN[i]:
                discrete_s.append(0)
            elif s[i] >= MAX[i]:
                discrete_s.append(N_CHUNKS[i] - 1)
            else:
                discrete_s.append(math.floor((s[i] - MIN[i]) * CHUNK_MULTIPLIER[i]))
        return tuple(discrete_s)

    @staticmethod
    def new_tables():
        # (number of chunks for s[0], ..., number of chunks for s[7], number of tables, number of possible coefficients)
        tables = np.zeros(N_CHUNKS + [N_TABLES, N_K])
        return tables

    def save_tables(self, save_path):
        # TODO save tables
        pass

    def load_tables(self, load_path):
        # TODO load tables
        return self.new_tables()

    def get_coefficients(self, s, eps):
        """Get coefficients for PID controllers.
        From Q-learning point of view, it just returns actions (one per table)
        Note: it returns generator, not a list"""
        s_d = self.discretize(s)
        if eps > 0 and random.random() < eps:
            k_indices = np.random.randint(low=N_K, size=N_TABLES)
        else:
            k_indices = self.tables[s_d].argmax(axis=1)

        coefficients = (K[i][k_indices[i]] for i in range(N_TABLES))

        self.prev_s_d = s_d
        self.prev_k_indices = k_indices

        return coefficients

    def update_tables(self, new_s, reward, lr, discount):
        """Compute new values for tables based on previous action"""
        if len(self.prev_s_d) == 0:
            print('Attempting to update tables without experience')
            return

        new_s_d = self.discretize(new_s)

        # update tables 1-3 and 4-6 based and reward from env
        prev_mask = np.zeros(shape=(N_TABLES, N_K), dtype=bool)
        for table_i in range(N_TABLES-3):  # exclude last 3 tables
            prev_mask[table_i, self.prev_k_indices[table_i]] = True

        prev_table_view = self.tables[self.prev_s_d]

        tmp1 = discount * self.tables[new_s_d][:6].max(axis=1)
        tmp2 = prev_table_view[prev_mask]
        td = lr * (reward + tmp1 - tmp2)
        prev_table_view[prev_mask] = prev_table_view[prev_mask] + td

        # update tables 7-9 based on rocket angle
        prev_mask = np.zeros(shape=(N_TABLES, N_K), dtype=bool)
        for table_i in range(6, N_TABLES):  # only last 3 tables
            prev_mask[table_i, self.prev_k_indices[table_i]] = True

        reward = -abs(new_s[4])  # reward for PID3 is based on rocket angle

        prev_table_view = self.tables[self.prev_s_d]

        tmp1 = discount * self.tables[new_s_d][6:].max(axis=1)
        tmp2 = prev_table_view[prev_mask]
        td = lr * (reward + tmp1 - tmp2)
        prev_table_view[prev_mask] = prev_table_view[prev_mask] + td

    # ==== PID + Q tables ====
    def get_actions(self, s, eps):
        """Compute action based on PID coefficients from Q-tables"""
        kp1, ki1, kd1, kp2, ki2, kd2, kp3, ki3, kd3 = self.get_coefficients(s, eps)
        dx, dy, vel_x, vel_y, theta, theta_dot, leg_contact_left, leg_contact_right = s

        Fe = self.pids[0].compute_output(min(abs(dx), 0.3)*0.4 - dy*0.2, kp1, ki1, kd1)
        Fs = self.pids[1].compute_output(theta*5, kp2, ki2, kd2)
        psi = self.pids[2].compute_output(theta + dx/5, kp3, ki3, kd3)

        return Fe, Fs, psi


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
