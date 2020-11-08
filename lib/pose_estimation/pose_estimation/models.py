import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

class NetworkParameters(object):
    def __init__(self, architecture, n_primitives, make_dense=False):
        self.architecture = architecture
        self.n_primitives = n_primitives
        self.make_dense = True
        print(architecture, n_primitives)

    @property
    def network(self):
        networks = dict(
            resnet18=ResNet18
        )
        return networks[self.architecture.lower()]

    def primitive_layer(self, n_primitives, input_channels):
        pose_module = Pose(n_primitives, self._build_modules(n_primitives, input_channels))
        return pose_module

    def _build_modules(self, n_primitives, input_channels):
        modules = {
            "translations": Translation(n_primitives, input_channels, self.make_dense),
            "rotations": Rotation(n_primitives, input_channels, self.make_dense),
        }
        return modules

class ResNet18(nn.Module):
    def __init__(self, network_params):
        super(ResNet18, self).__init__()
        self._network_params = network_params

        self._features_extractor = models.resnet18(pretrained=True)
        self._features_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        self._features_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._primitive_layer = self._network_params.primitive_layer(
            self._network_params.n_primitives,
            512
        )

    def forward(self, X):
        X = X.float() / 255.0
        x = self._features_extractor(X)
        return self._primitive_layer(x.view(-1, 512, 1, 1, 1))

class Translation(nn.Module):
    """A layer that predicts the translation vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Translation, self).__init__()
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

class Rotation(nn.Module):
    """A layer that predicts the rotation vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Rotation, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the 4 quaternions of each primitive, namely
        # BxMx4
        self._rotation_layer = nn.Conv3d(
            input_channels, self._n_primitives*4, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the 4 parameters of the quaternion for every primitive
        # and add a non-linearity as L2-normalization to enforce the unit
        # norm constrain
        quats = self._rotation_layer(X)[:, :, 0, 0, 0]
        quats = quats.view(-1, self._n_primitives, 4)
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        rotations = rotations.view(-1, self._n_primitives*4)

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
        for i, m in enumerate(self._pose_params.values()):
            self.add_module("layer%d" % (i,), m)

    def forward(self, X):
        translations = self._pose_params["translations"].forward(X)
        rotations = self._pose_params["rotations"].forward(X)

        return PoseParameters(
            translations, rotations
        )

def train_on_batch(
    model,
    optimizer,
    loss_fn,
    X,
    y_target,
    device
):
    # Zero the gradient's buffer
    optimizer.zero_grad()
    y_hat = model(X)
    options = {"device": device}

    loss, debug_stats = loss_fn(
        y_hat,
        y_target,
        options
    )

    # Do the backpropagation
    loss.backward()

    # TODO?
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    # Do the update
    optimizer.step()

    return (
        loss.item(),
        [x.data if hasattr(x, "data") else x for x in y_hat],
        debug_stats
    )

def optimizer_factory(args, model):
    """Based on the input arguments create a suitable optimizer object
    """
    params = model.parameters()

    if args.optimizer == "SGD":
        return optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        return optim.Adam(
            params,
            lr=args.lr
        )
