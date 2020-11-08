#!/usr/bin/env python
"""Script used to perform a forward pass using a previously trained model and
visualize the corresponding primitives
"""
import argparse
import os
import sys

import numpy as np
import torch
import math

from pyquaternion import Quaternion

from utils import store_primitive_parameters, load_all_primitive_parameters, \
    load_pose, rotation_matrix_to_quaternion, find_files, pose_to_string, string_to_pose, \
    quat_pose_to_string
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
        help="Path to the directory containing the poses of the instance"
    )

    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )

    parser.add_argument(
        "--labels_filename",
        type=str,
        default="labels_quat.txt",
        help="Name of the labels file"
    )

    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold"
    )

    args = parser.parse_args(argv)
    primitives = load_all_primitive_parameters(args.primitives_directory, args.prob_threshold)
    paths = find_files(args.pose_directory + "/**meta.txt")
    
    # Serializes poses in sorted order
    with open(os.path.join(args.output_directory, args.labels_filename), 'w+') as labels_file: 
        # Loop over every meta file
        for pose_path in sorted(paths):
            # print(pose_path)
            _, rt, _ = load_pose(pose_path)
            
            q_obj = rotation_matrix_to_quaternion(rt[:3,:3])
            

            M = len(primitives)
            M_poses = np.zeros(shape=(M, 7))

            i = 0
            for prim in primitives:
                prim["obj_pose"] = q_obj
                pose_matrix = superquadric_pose(prim["obj_pose"], prim["rotation"], prim["location"])
                quat = rotation_matrix_to_quaternion(pose_matrix[:3, :3]).elements
                tran = pose_matrix[:3, 3]
                M_poses[i, :4] = quat[:]
                M_poses[i, 4:] = tran[:]
                i += 1


            # Serialize
            to_write = quat_pose_to_string(M_poses)
            labels_file.write(to_write)
            labels_file.write("\n")
            # if not (string_to_pose(to_write) == M_poses).all():
            #     import pdb
            #     pdb.set_trace()
            assert(not np.isnan(M_poses.any()))
            assert((string_to_pose(to_write) == M_poses).all())

    labels_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
