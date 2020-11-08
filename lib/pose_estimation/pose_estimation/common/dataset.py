import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Normalize as TorchNormalize

class BaseDataset(Dataset):
    """Dataset is a wrapper for all datasets we have
    """
    def __init__(self, dataset_object, datafactory, transform=None):
        """
        Arguments:
        ---------
            dataset_object: a dataset object that can be either Instance or Category
            datafactory: a factory that creates objects that wrap a data point
            transform: Callable that applies a transform to a sample
        """
        self._dataset_object = dataset_object
        print("{} datapoints in total ...".format(len(self._dataset_object)))

        # Get the voxelizer to be used
        self._creator = datafactory.creator
        
        # Operations on the datapoints
        self.transform = transform

        # Data shape
        self._input_dim = self._creator.input_shape
        self._output_dim = self._creator.output_shape

    def __len__(self):
        return len(self._dataset_object)

    def __getitem__(self, idx):
        (X, y_target) = self._creator.get_Xy(self._dataset_object, idx)

        datapoint = (
            X,
            y_target.astype(np.float32)
        )

        # Store the dimentionality of the input tensor and the y_target tensor
        self._input_dim = datapoint[0].shape
        self._output_dim = datapoint[1].shape

        if self.transform:
            datapoint = self.transform(datapoint)

        return datapoint

    def get_random_datapoint(self, idx=None):
        if idx is None:
            idx = np.random.choice(np.arange(self.__len__()))
        return self.__getitem__(idx)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        X, y_target = sample

        # Do some sanity checks to ensure that the inputs have the appropriate
        # dimensionality
        # assert len(X.shape) == 4
        return (torch.from_numpy(X), torch.from_numpy(y_target).float())

class Normalize(object):
    """Normalize image based based on ImageNet."""
    def __call__(self, sample):
        X, y_target = sample
        X = X.float()

        # The normalization will only affect X
        normalize = TorchNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        X = X.float() / 255.0
        
        return (normalize(X), y_target)

def compose_transformations(datafactory):
    transformations = [ToTensor()]
    if datafactory == "image":
        transformations.append(Normalize())

    return transforms.Compose(transformations)

def get_dataset_type(loss_type):
    return {
        "matrix_loss": BaseDataset
    }[loss_type]
