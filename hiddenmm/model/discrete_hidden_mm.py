import numpy as np

import hiddenmm.model.markov_chain as mc
import hiddenmm.constants as constants


class DiscreteHiddenMM:
    """ Class implementing a discrete Hidden Markov Model """

    def __init__(self, markov_chain: mc.MarkovChain, projection: np.ndarray):
        self.markov_chain = markov_chain
        self.projection = projection
        self.num_outputs = self.projection.shape[1]

        if self.markov_chain.num_states != projection.shape[0]:
            raise ValueError("Dimension mismatch: projection dies not match number of states of Markov Chain")

        if not np.all((self.projection.sum(axis=1) - 1.0) < constants.EPSILON):
            raise ValueError("Projection matrix distribution does not sum up to one.")

        if not np.all(self.projection >= 0.0):
            raise ValueError("Projection matrix distribution must be positive.")

    def generate(self, n: int) -> np.ndarray:
        """ Generate given markov model """
        underlying_chain = self.markov_chain.generate(n)

        result = []

        for i in range(n):
            result.append(np.random.choice(self.num_outputs, p=self.projection[underlying_chain[i]]))

        return np.vstack([underlying_chain, np.array(result, dtype=int)]).T

    def likelihood(self, observations: np.ndarray):
        """ Return likelihood of supplied observation series """
        dimension = observations.shape[0]

        if dimension > 0:
            vector = self.markov_chain.initial * self.projection[:, observations[0]]

            for i in range(1, dimension):
                vector = (vector @ self.markov_chain.transition_matrix) * self.projection[:, observations[i]]

            return vector.sum()
        else:
            return 0.0
