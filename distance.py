
import numpy as np


def distance(a, b):
    return np.linalg.norm(np.subtract(a, b))
