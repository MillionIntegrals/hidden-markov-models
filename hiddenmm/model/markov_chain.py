import numpy as np

import hiddenmm.constants as constants


class MarkovChain:
    """ Basic markov chain generative description consisting of initial state and transition matrix """

    def __init__(self, initial, transition_matrix):
        self.initial = initial
        self.transition_matrix = transition_matrix
        self.num_states = self.initial.shape[0]

        if self.initial.shape[0] != self.transition_matrix.shape[0]:
            raise ValueError("Initial state shape {} does not match transition matrix shape {}".format(
                self.initial.shape[0], self.transition_matrix.shape[0]
            ))

        if self.initial.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("Initial state shape {} does not match transition matrix shape {}".format(
                self.initial.shape[0], self.transition_matrix.shape[1]
            ))

        initial_sum = np.sum(self.initial)

        if not np.abs(initial_sum - 1.0) < constants.EPSILON:
            raise ValueError("Initial distribution does not sum up to one: {}".format(initial_sum))

        if not np.all(self.initial >= 0.0):
            raise ValueError("Initial distribution parameters must be positive.")

        if not np.all(np.abs(self.transition_matrix.sum(axis=1) - 1.0) < constants.EPSILON):
            raise ValueError("Transition matrix distributions do not sum up to one.")

        if not np.all(self.transition_matrix >= 0.0):
            raise ValueError('Transition matrix parameters must be positive.')

    def generate(self, n):
        """ Generate a run of length n of given markov chain """
        result = []

        if n > 0:
            current_value = np.random.choice(self.num_states, p=self.initial)
            result.append(current_value)

            for idx in range(n - 1):
                current_value = np.random.choice(self.num_states, p=self.transition_matrix[current_value])
                result.append(current_value)

            return np.array(result, dtype=int)
        else:
            return np.array([], dtype=int)
