import numpy as np


def rescale_vector(vector: np.ndarray) -> np.ndarray:
    """ Rescale vector so that it sums up to 1, often used here for numerical stability reasons """
    total = vector.sum()

    if np.abs(total) > 1e-10:
        return vector / total
    else:
        return vector
