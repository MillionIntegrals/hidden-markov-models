import numpy as np

import hiddenmm.model.markov_chain as mcmodule
import hiddenmm.constants as constants


class DiscreteHiddenMM:
    """ Class implementing a discrete Hidden Markov Model """

    def __init__(self, mc: mcmodule.MarkovChain, projection: np.ndarray):
        self.mc = mc
        self.projection = projection

        if self.mc.num_states != projection.shape[0]:
            raise ValueError("Dimension mismatch: projection dies not match number of states of Markov Chain")

        if not np.all((self.projection.sum(axis=1) - 1.0) < constants.EPSILON):
            raise ValueError("Projection matrix distribution does not sum up to one.")

        if not np.all(self.projection >= 0.0):
            raise ValueError("Projection matrix distribution must be positive.")
