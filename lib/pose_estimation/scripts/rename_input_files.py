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
        "input_directory",
        help="Path to the directory containing the poses of the instance"
    )
    
    args = parser.parse_args(argv)
    paths = sorted(find_files(args.input_directory + "/**color.png"))
    print(paths)

    import os
    for i in range(0, len(paths)):
        os.rename(paths[i], os.path.join(args.input_directory, str(i) + ".png"))

if __name__ == "__main__":
    main(sys.argv[1:])
