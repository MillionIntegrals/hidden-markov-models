import numpy as np
import nose.tools as ntools

import hiddenmm.model.markov_chain as mc
import hiddenmm.model.discrete_hidden_mm as dhmm


def test_successful_creation():
    pi = np.array([0.1, 0.9])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    chain = mc.MarkovChain(pi, a)

    b = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.2, 0.3, 0.3]
    ])

    dhmm.DiscreteHiddenMM(chain, b)


@ntools.raises(ValueError)
def test_dimension_mismatch():
    pi = np.array([0.1, 0.9])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    chain = mc.MarkovChain(pi, a)

    b = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.2, 0.3, 0.3],
        [0.1, 0.1, 0.2, 0.3, 0.3]
    ])

    dhmm.DiscreteHiddenMM(chain, b)


@ntools.raises(ValueError)
def test_distribution_does_not_sum_up():
    pi = np.array([0.1, 0.9])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    chain = mc.MarkovChain(pi, a)

    b = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.2, 0.3, 0.5]
    ])

    dhmm.DiscreteHiddenMM(chain, b)


@ntools.raises(ValueError)
def test_distribution_negative():
    pi = np.array([0.1, 0.9])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    chain = mc.MarkovChain(pi, a)

    b = np.array([
        [0.2, 0.2, 0.2, 0.6, -0.2],
        [0.1, 0.1, 0.2, 0.3, 0.3],
    ])

    dhmm.DiscreteHiddenMM(chain, b)
