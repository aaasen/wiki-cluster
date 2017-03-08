
import random
from k_means import _sample, _normalize
import numpy as np

random.seed(0)

array = _normalize(np.random.rand(10))
samples = np.array([0 for x in array])


for i in range(10000):
    samples[_sample(array)] += 1


print(array - _normalize(samples))
print(array)
print(_normalize(samples))
