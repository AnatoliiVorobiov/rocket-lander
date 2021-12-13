import math
import random

import numpy as np


S_SIZE = 8  # number of observations in a state
N_CONTROLLERS = 3
N_TABLES = N_CONTROLLERS * 3

N_CHUNKS = [5, 5, 5, 5, 5, 5, 2, 2]
MIN = [-1, -1, 0, 0, -1, -1, 0, 0]
MAX = [1, 1, 5, 5, 1, 1, 1, 1]
CHUNK_MULTIPLIER = list(N_CHUNKS[i] / (MAX[i] - MIN[i]) for i in range(S_SIZE))

K = [0, 0.1, 0.5, 1, 3, 6, 10]


class QPIDAgent:
    def __init__(self, load_path=None):
        if load_path is None:
            self.tables = self.new_tables()
        else:
            self.tables = self.load_tables(load_path)

        self.prev_s = []
        self.prev_k_indices = []

    # ==== Q tables ====
    @staticmethod
    def discretize(s):
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

    def update_tables(self, new_s, reward):
        if len(self.prev_s) == 0:
            print('Attempting to update tables without experience')
            return

        new_s = self.discretize(new_s)
        # TODO

    # ==== PID + Q tables ====
    def get_actions(self, s, eps):
        # TODO use get_coefficients to compute PID output
        pass
