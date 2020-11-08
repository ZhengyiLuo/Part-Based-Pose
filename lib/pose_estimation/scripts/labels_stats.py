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
        "pose_directory",
        help="Path to the directory containing the poses file"
    )

    args = parser.parse_args(argv)
    
    with open(os.path.join(args.pose_directory, "outputs/labels.txt"), 'r') as fp:
        label = fp.readline()
        min_trans = float("inf")
        max_trans = float("-inf")

        min_rot = float("inf")
        max_rot = float("-inf")
        while label:
            
            pose = string_to_pose(label.split("\n", 1)[0])
            pose = np.reshape(pose, (-1, 3, 4))

            trans = pose[:, :3, 3]
            
            #print("trans:", np.min(trans), np.max(trans))
            if max_trans < np.max(trans):
                max_trans = np.max(trans)

            if min_trans > np.min(trans):
                min_trans = np.min(trans) 

            rot = pose[:, :3, :3]

            if max_rot < np.max(rot):
                max_rot = np.max(rot)

            if min_rot > np.min(rot):
                min_rot = np.min(rot) 

            #print("rot:", np.min(rot), np.max(rot))
            label = fp.readline()
        print("Trans bounds:", min_trans, max_trans)
        print("Rot bounds:", min_rot, max_rot)
                

if __name__ == "__main__":
    main(sys.argv[1:])
