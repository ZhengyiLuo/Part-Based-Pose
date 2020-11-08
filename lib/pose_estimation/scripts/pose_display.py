import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from pyquaternion import Quaternion

from utils import get_colors, store_primitive_parameters, load_all_primitive_parameters
from visualization_utils import points_on_sq_surface, save_params_as_ply \

from arguments import add_datatype_parameters, add_nn_parameters, \
    add_dataset_parameters, add_training_parameters, data_input_shape
from output_logger import get_logger

from pose_estimation.common.dataset import get_dataset_type, \
    compose_transformations
from pose_estimation.common.model_factory import DatasetBuilder
from pose_estimation.common.batch_provider import BatchProvider
from pose_estimation.models import NetworkParameters, train_on_batch, \
    optimizer_factory
from pose_estimation.loss_functions import matrix_loss
from pose_estimation.datafactory import DataFactory

# from mayavi import mlab

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

M = 11
data_output_shape = (M, 7)
architecture = "resnet18"
device = torch.device("cuda:0")
weight_file = "../trained_models/model_10"
primitives_directory = "../superquadric_primitives/chair1"
dataset_directory = "/hdd/zen/data/Reallite/chair_reallite1/chair_ycb/data/0000/"
prob_threshold = 0.5


network_params = NetworkParameters(architecture, M, False)
model = network_params.network(network_params)
# Move model to device to be used
model.to(device)
if weight_file is not None:
    # Load the model parameters of the previously trained model
    model.load_state_dict(
        torch.load(weight_file)
    )
    print("Loading...", weight_file)
model.eval()

# Keep track of the files containing the parameters of each primitive
primitives = load_all_primitive_parameters(primitives_directory, prob_threshold)
gt_primitives = list(primitives)
colors = get_colors(M)

parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
add_nn_parameters(parser)
add_dataset_parameters(parser)
add_datatype_parameters(parser)
add_training_parameters(parser)
args = parser.parse_args("")
print(args)

data_type = "image"
data_factory = DataFactory(
        data_type,
        tuple([data_input_shape(args), data_output_shape])
    )
dataset = get_dataset_type("matrix_loss")(
        (DatasetBuilder()
            .with_dataset(args.dataset_type)
            .build(dataset_directory)),
        data_factory,
        transform=compose_transformations(data_type)
    )
dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
primitive_list = []
total_runs = 0

i = 0
for sample in dataloader:
    total_runs +=1
    X, y_target = sample

    # Show input image
    img = X.numpy()[0]
    img = np.transpose(img, (1,2,0))
    img = img.reshape((224, 224, 3))
    imgplot = plt.imshow(img)
    plt.show()

    X, y_target = X.to(device), y_target.to(device)

    # Declare some variables
    B = y_target.shape[0]  # batch size
    M = y_target.shape[1]  # number of primitives
    poses_target = y_target.view(B, M, 7).detach().cpu().numpy()
    rotations_target = poses_target[:, :, :4].reshape(B, M, 4)[0]
    translations_target = poses_target[:, :, 4:].reshape(B, M, 3)[0]

    # # Do the forward pass
    y_hat = model(X)
    translations = y_hat[0].detach().cpu().numpy().reshape(B, M, 3)[0]
    rotations = y_hat[1].detach().cpu().numpy().reshape(B, M, 4)[0]
    
    

    plt.show()    
    
    for p in primitives:
        # primitives[i]["rotation"] = rotations[i]
        # primitives[i]["location"] = translations[i]

        # gt_primitives[i]["rotation"] = rotations_target[i]
        # gt_primitives[i]["location"] = translations_target[i]

        x_tr, y_tr, z_tr, prim_pts =\
            points_on_sq_surface(
                p["size"][0],
                p["size"][1],
                p["size"][2],
                p["shape"][0],
                p["shape"][1],
                Quaternion(rotations[i]).rotation_matrix.reshape(3, 3),
                np.array(translations[i]).reshape(3, 1),
                p["tapering"][0],
                p["tapering"][1],
                None,
                None
            )
        primitive_list.append((prim_pts, p['color']))
        
#         break

    break

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for prim_pts, color in primitive_list:
    ax.scatter(prim_pts[0,:], prim_pts[1,:], prim_pts[2,:], c =color)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_ylim(-0.5, 0.5)
ax.set_xlim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
plt.show()