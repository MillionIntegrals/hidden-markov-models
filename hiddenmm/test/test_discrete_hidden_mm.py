import numpy as np
import nose.tools as ntools

import hiddenmm.model.markov_chain as mc
import hiddenmm.model.discrete_hidden_mm as dhmm
import hiddenmm.constants as cnst


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


def test_generation_1():
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

    model = dhmm.DiscreteHiddenMM(chain, b)

    result = model.generate(100)

    assert result.shape[0] == 100
    assert result.shape[1] == 2


def test_likelihood_summation():
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

    model = dhmm.DiscreteHiddenMM(chain, b)

    summation = 0.0

    for i in range(5):
        for j in range(5):
            for k in range(5):
                for z in range(5):
                    observations = [i, j, k, z]

                    likelihood = model.likelihood(np.array(observations, dtype=int))

                    summation += likelihood

    assert np.abs(summation - 1.0) < cnst.EPSILON
