from typing import List, Type
import dataclasses
from .. import NormWrapper
import numpy as np

'''
'''


class BinarySVM:
    def __init__(self, input_dim: int):
        if input_dim < 0:
            raise ValueError("Model must have input dimension >= 0")
        self._INPUT_DIM = input_dim
        self._weights = [0] * self._INPUT_DIM  # this is our 'b'

    def train_naive_runtime(self, x_t: List[List[float]], y_t: List[float]):

        x_tt = [x_i + [1] for x_i in x_t]

        x_train = np.array([np.array(x_i) for x_i in x_tt])
        y_train = np.array(y_t)

        self._weights = self._normw.get_minimisers(self._normw, x_train, y_train)
        print(self._weights)

    def _get_weight_i(self, index: int):
        if index < 0 or index > self._INPUT_DIM - 1:
            raise IndexError("Index out of bounds")
        return self._weight[index]

    def _get_intercept(self):
        return self._weight[-1]
