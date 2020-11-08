import numpy as np
import os
import uuid

class DataFactory(object):
    def __init__(self, dataset_type, data_shape):
        if not isinstance(dataset_type, str) or\
               dataset_type not in ["image"]:
                    raise AttributeError("The data type is invalid")

        self._dataset_type = dataset_type
        self.data_shape = data_shape

    @property
    def creator(self):
        if self._dataset_type == "image":
            return PrecomputedData(self.data_shape)

class PrecomputedData(object):
    """PrecomputedData is a wrapper for precomputed data
    """
    def __init__(self, data_shape):
        self._data_shape = data_shape

    @property
    def input_shape(self):
        return self._data_shape[0]

    @property
    def output_shape(self):
        return self._data_shape[1]

    def get_Xy(self, dataset, idx):
        (X, y) = dataset.get_datapoint(idx)
        # Transpose image to have the right size for pytorch
        return (np.transpose(X, (2, 0, 1)), y)
        