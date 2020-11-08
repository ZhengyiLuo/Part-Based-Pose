import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from lib.models.pose_models import *

class Prim_Rotation(nn.Module):
    """A layer that predicts the rotation vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False, rot_repr = "quat"):
        super(Prim_Rotation, self).__init__()
        self._n_primitives = n_primitives
        self._rot_repr = rot_repr

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the 4 quaternions of each primitive, namely
        # BxMx4
        if self._rot_repr == "quat":
            self._rotation_layer = nn.Conv3d(
                input_channels, self._n_primitives*4, 1
            )
        elif self._rot_repr == "6d":
            self._rotation_layer = nn.Conv3d(
                input_channels, self._n_primitives*6, 1
            )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the 4 parameters of the quaternion for every primitive
        # and add a non-linearity as L2-normalization to enforce the unit
        # norm constrain
        if self._rot_repr == "quat":
            quats = self._rotation_layer(X)[:, :, 0, 0, 0]
            quats = quats.view(-1, self._n_primitives, 4)
            rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
            rotations = rotations.view(-1, self._n_primitives*4)
        else: 
            rots = self._rotation_layer(X)[:, :, 0, 0, 0]
            rots = rots.view(-1, self._n_primitives, 6)
            rotations = rots.view(-1, self._n_primitives*6)

        return rotations

class PoseParameters(object):
    """Represents the \lambda_m."""
    def __init__(self, translations, rotations):
        self.translations = translations
        self.rotations = rotations

        # Check that everything has a len(shape) > 1
        for x in self.members[:-2]:
            assert len(x.shape) > 1

    def __getattr__(self, name):
        if not name.endswith("_r"):
            raise AttributeError()

        prop = getattr(self, name[:-2])
        if not torch.is_tensor(prop):
            raise AttributeError()

        return prop.view(self.batch_size, self.n_primitives, -1)

    @property
    def members(self):
        return (
            self.translations,
            self.rotations
        )

    @property
    def batch_size(self):
        return self.probs.shape[0]

    @property
    def n_primitives(self):
        return self.probs.shape[1]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        return self.members[i]


class Pose(nn.Module):
    def __init__(self, n_primitives, pose_params):
        super(Pose, self).__init__()
        self._n_primitives = n_primitives
        self._pose_params = pose_params
        self._update_params()

    def _update_params(self):
        self.add_module("layer%d" % (0,), self._pose_params["rotations"])
        self.add_module("layer%d" % (1,), self._pose_params["translations"])

    def forward(self, X):
        translations = self._pose_params["translations"].forward(X)
        rotations = self._pose_params["rotations"].forward(X)

        return PoseParameters(
            translations, rotations
        )


class Prim_Translation(nn.Module):
    """A layer that predicts the translation vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Prim_Translation, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the translation vector of each primitive, namely
        # BxMx3
        self._translation_layer = nn.Conv3d(
            input_channels, self._n_primitives*3, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the BxM*3 translation vectors for every primitive and ensure
        # that they lie inside the unit cube
        translations = torch.tanh(self._translation_layer(X)) * 0.51

        return translations[:, :, 0, 0, 0]
