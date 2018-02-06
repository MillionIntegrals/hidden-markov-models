import numpy as np

import hiddenmm.constants as cnst


def rescale_vector(vector: np.ndarray) -> (np.ndarray, float):
    """ Rescale vector so that it sums up to 1, often used here for numerical stability reasons """
    total = vector.sum()

    if np.abs(total) > cnst.SMALL_EPSILON:
        return vector / total, 1.0 / total
    else:
        return vector, 1.0
