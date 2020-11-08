#!/usr/bin/env python
"""Script used to perform a forward pass using a previously trained model and
visualize the corresponding primitives
"""
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

def display_primitives(primitive_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    transform_world_to_matplot = np.array([
        [1,0,0],
        [0,0,1],
        [0,-1,0]
    ])
    print(len(primitive_list))
    for prim_pts, color in primitive_list:
        prim_pts = np.matmul(prim_pts.T, transform_world_to_matplot)
        ax.scatter(prim_pts[:,0], prim_pts[:,1], prim_pts[:,2], c =color)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    plt.show()
    
    

def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )

    parser.add_argument(
        "primitives_directory",
        help="Path to the directory containing the superquadrics of the instance"
    )

    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    
    parser.add_argument(
        "--weight_file",
        default=None,
        help="The path to the previously trainined model to be used"
    )

    parser.add_argument(
        "--save_prediction_as_mesh",
        action="store_true",
        help="When true store prediction as a mesh"
    )

    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )

    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold"
    )

    # Parse args
    add_nn_parameters(parser)
    add_dataset_parameters(parser)
    add_datatype_parameters(parser)
    add_training_parameters(parser)
    args = parser.parse_args(argv)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda:0")
    print("Running code on ", device)
    
    # TODO
    M = 11
    data_output_shape = (M, 7)

    # Create a factory that returns the appropriate data type based on the
    # input argument
    data_factory = DataFactory(
        args.data_type,
        tuple([data_input_shape(args), data_output_shape])
    )

    # Create a dataset instance to generate the samples for training
    dataset = get_dataset_type("matrix_loss")(
        (DatasetBuilder()
            .with_dataset(args.dataset_type)
            .build(args.dataset_directory)),
        data_factory,
        transform=compose_transformations(args.data_type)
    )
    
    # TODO: Change batch_size in dataloader
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    network_params = NetworkParameters(args.architecture, M, False)
    model = network_params.network(network_params)
    # Move model to device to be used

    model.to(device)    
    if args.weight_file is not None:
        # Load the model parameters of the previously trained model
        model.load_state_dict(
            torch.load(args.weight_file)
        )
    model.eval()


    # Keep track of the files containing the parameters of each primitive
    primitives = load_all_primitive_parameters(args.primitives_directory, args.prob_threshold)
    gt_primitives = list(primitives)
    colors = get_colors(M)

    # Prepare matlab figs 
    # mlab.view(azimuth=0.0, elevation=0.0, distance=2)
    
    # Iterate thru the data
    total_runs = 0
    total = 0
    r_loss_total = 0
    t_loss_total = 0
    # fp = open(os.path.join(args.output_directory, "stats.csv"), "w")
    # fp.write("loss_total\trot_loss\ttrans_loss\t\n")

    for sample in dataloader:
        primitive_list = []
        total_runs +=1
        X, y_target = sample
        
        # Show input image
        # img = X.numpy()[0]
        # img = np.transpose(img, (1,2,0))
        # img = img.reshape((224, 224, 3))
        # imgplot = plt.imshow(img)
        # plt.show()

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

        # Loss computations
        # options = dict()
        # options["device"] = device
        # loss, extra = matrix_loss(y_hat, y_target, options)
        # total += (extra["r_loss"] + extra["t_loss"])
        # r_loss_total += extra["r_loss"]
        # t_loss_total += extra["t_loss"]

        # fp.write(str(total / total_runs))
        # fp.write("\t")
        # fp.write(str(r_loss_total / total_runs))
        # fp.write("\t")
        # fp.write(str(t_loss_total / total_runs))
        # fp.write("\t")
        # fp.write("\n")
        
        # if total_runs % 50 == 0:
        #     print(total / total_runs )

        i = 0
        # fig1 = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        # fig2 = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))

        for p in primitives:
            # primitives[i]["rotation"] = rotations[i]
            # primitives[i]["location"] = translations[i]

            # gt_primitives[i]["rotation"] = rotations_target[i]
            # gt_primitives[i]["location"] = translations_target[i]
            print("using GT...")
            x_tr, y_tr, z_tr, prim_pts =\
                points_on_sq_surface(
                    p["size"][0],
                    p["size"][1],
                    p["size"][2],
                    p["shape"][0],
                    p["shape"][1],
                    # Quaternion(rotations_target[i]).rotation_matrix.reshape(3, 3),
                    # np.array(translations_target[i]).reshape(3, 1),
                    Quaternion(rotations[i]).rotation_matrix.reshape(3, 3),
                    np.array(translations[i]).reshape(3, 1),
                    p["tapering"][0],
                    p["tapering"][1],
                    None,
                    None
                )
            
            primitive_list.append((prim_pts, p['color']))
            i += 1

        print("-------- GT ---------")    
        print(rotations_target)
        print(translations_target)
        print("--------- Pred ---------")
        print(rotations)
        print(translations)
        display_primitives(primitive_list)
            # x_tr, y_tr, z_tr, prim_pts =\
            #     points_on_sq_surface(
            #         p["size"][0],
            #         p["size"][1],
            #         p["size"][2],
            #         p["shape"][0],
            #         p["shape"][1],
            #         Quaternion(rotations_target[i]).rotation_matrix.reshape(3, 3),
            #         np.array(translations_target[i]).reshape(3, 1),
            #         p["tapering"][0],
            #         p["tapering"][1],
            #         None,
            #         None,
            #     )
        # break
            

            
 


if __name__ == "__main__":
    main(sys.argv[1:])
