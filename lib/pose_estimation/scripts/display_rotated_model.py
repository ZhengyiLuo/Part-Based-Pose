#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
import torch
import math

from pyquaternion import Quaternion

from utils import get_colors, store_primitive_parameters, load_all_primitive_parameters, \
    load_pose, rotation_matrix_to_quaternion, find_files, pose_to_string, string_to_pose
from visualization_utils import points_on_sq_surface, points_on_cuboid, \
    save_prediction_as_ply, save_params_as_ply, superquadric_pose

def main(argv):
    parser = argparse.ArgumentParser(
        description="Load in a pose and transform superquadrics"
    )

    parser.add_argument(
        "primitives_directory",
        help="Path to the directory containing the superquadrics of the instance"
    )

    parser.add_argument(
        "pose_directory",
        help="Path to the directory containing the poses file"
    )

    parser.add_argument(
        "pose_index",
        type=int,
        default=0,
        help="Index of the pose to be displayed"
    )

    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold"
    )

    args = parser.parse_args(argv)
    primitives = load_all_primitive_parameters(args.primitives_directory, args.prob_threshold)
    paths = find_files(args.pose_directory + "meta/**meta.txt")

    # Loop over every meta file
    pose_path = sorted(paths)[0]
    _, rt, _ = load_pose(pose_path)
    q_obj = rotation_matrix_to_quaternion(rt[:3,:3])

    M = len(primitives)
    M_poses = np.zeros(shape=(M, 3, 4))

    i = 0
    for prim in primitives:
        prim["obj_pose"] = q_obj
    
    print(args.pose_directory)
    with open(os.path.join(args.pose_directory, "outputs/labels.txt"), 'r') as fp:
        for i in np.arange(0, args.pose_index + 1):
            label = fp.readline()
            if i == args.pose_index:
                pose = string_to_pose(label.split("\n", 1)[0])
                pose = np.reshape(pose, (-1, 3, 4))
                
                for m in np.arange(0, np.shape(pose)[0]):
                    primitives[m]["transform"] = pose[m]

    save_params_as_ply(os.path.join("../results/test/", "primitives.ply"), primitives)

if __name__ == "__main__":
    main(sys.argv[1:])
