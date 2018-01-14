import numpy as np
import hiddenmm.model.markov_chain as mc
import hiddenmm.constants as constants

import nose.tools as ntools


def test_creation_passes():
    """ Simplistic test case to make sure setup works """
    pi = np.array([0.1, 0.9])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    mc.MarkovChain(pi, a)


@ntools.raises(ValueError)
def test_creation_dim_fails():
    """ Dimension mismatch throws exception """
    pi = np.array([0.1, 0.9, 0.5])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    mc.MarkovChain(pi, a)


@ntools.raises(ValueError)
def test_creation_distribution_sum():
    """ Dimension mismatch throws exception """
    pi = np.array([0.1, 0.1])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    mc.MarkovChain(pi, a)


@ntools.raises(ValueError)
def test_creation_distribution_sum_1():
    """ Dimension mismatch throws exception """
    pi = np.array([0.1, 0.9])

    a = np.array([
        [0.1, 0.1],
        [0.5, 0.5]
    ])

    mc.MarkovChain(pi, a)


@ntools.raises(ValueError)
def test_creation_distribution_sum_2():
    """ Dimension mismatch throws exception """
    pi = np.array([0.1, 0.9])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.4]
    ])

    mc.MarkovChain(pi, a)


@ntools.raises(ValueError)
def test_creation_negative_distribution():
    """ Dimension mismatch throws exception """
    pi = np.array([1.1, -0.1])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.4]
    ])

    mc.MarkovChain(pi, a)


def test_simple_generation():
    """ Simple generation """
    pi = np.array([0.1, 0.9])

    a = np.array([
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    model = mc.MarkovChain(pi, a)

    result = model.generate(n=100)

    assert result.shape[0] == 100
    assert result.dtype == int


def test_constant_generation():
    """ Simple generation """
    pi = np.array([1.0, 0.0, 0.0])

    a = np.array([
        [1.0, 0.0, 0.0],
        [0.3, 0.4, 0.3],
        [0.1, 0.8, 0.1]
    ])

    model = mc.MarkovChain(pi, a)

    result = model.generate(n=100)

    assert np.all(result < constants.EPSILON)
