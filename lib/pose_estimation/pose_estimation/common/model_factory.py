import numpy as np
import re
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
import os
import pickle
from PIL import Image

import torch


from utils import string_to_pose

class InstanceModel(object):
    def __init__(self, base_dir):
        # Only 1 instance
        self.images_dir = os.path.join(base_dir, "inputs")
        # self.images_dir = base_dir
        self.poses_dir = os.path.join(base_dir, "outputs")
        self._datapoints = []
        self.datapoints

    def __len__(self):
        return self._n_datapoints()

    def _n_datapoints(self):
        return len(os.listdir(self.images_dir))

    @property
    def datapoints(self):
        if not self._datapoints:
            with open(os.path.join(self.poses_dir, "labels_quat.txt"), "r") as fp:
                for path_to_img in sorted(os.listdir(self.images_dir)):
                    if path_to_img.endswith((".jpg", ".png")) and "color" in path_to_img:
                        image_path = os.path.join(self.images_dir, path_to_img)
                        
                        # Grab pose of image
                        label = fp.readline()
                        pose = string_to_pose(label.split("\n", 1)[0])

                        M = np.shape(pose)[0]
                        rotation = pose[:, :4]
                        translation = pose[:, 4:]

                        pose = np.zeros((M, 7))
                        pose[:, :4] = rotation
                        pose[:, 4:] = translation

                        # Create datapoint
                        self._datapoints.append(
                            tuple([np.array(Image.open(image_path).convert("RGB")), pose])       
                        )
        return self._datapoints

    @property
    def random_datapoint(self):
        ri = np.random.choice(len(self.datapoints))
        return self.datapoints[ri]
    
    def get_datapoint(self, idx):
        if idx > self._n_datapoints() or idx < 0:
            print(idx)
            print("Tried to get datapoint at wrong idx.. exiting.")
            exit(1)
        return self.datapoints[idx]

def model_factory(dataset_type):
    return {
        "instance": InstanceModel
    }[dataset_type]

class DatasetBuilder(object):
    def __init__(self):
        self._dataset_class = None

    def with_dataset(self, dataset_type):
        self._dataset_class = model_factory(dataset_type)
        return self

    def build(self, base_dir):
        dataset = self._dataset_class(base_dir)

        return dataset
