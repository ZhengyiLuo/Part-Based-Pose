import pandas as pd
import numpy as np
import pickle
import glob
import math
from pyquaternion import Quaternion

from functools import reduce

def get_colors(M):
    import seaborn as sns
    sns.set()
    return sns.color_palette("Paired")

import json

def parse_train_test_splits(train_test_splits_file, model_tags):
    splits = {}
    if not train_test_splits_file.endswith("csv"):
        raise Exception("Input file %s is not csv" % (train_test_splits_file,))
    df = pd.read_csv(
        train_test_splits_file,
        names=["id", "synsetId", "subSynsetId", "modelId", "split"]
    )
    keep_from_model = reduce(
        lambda a, x: a | (df["synsetId"] in x),
        model_tags,
        False
    )
    # Keep only the rows from the model we want
    df_from_model = df[keep_from_model]

    train_idxs = df_from_model["split"] == "train"
    splits["train"] = df_from_model[train_idxs].modelId.values.tolist()
    test_idxs = df_from_model["split"] == "test"
    splits["test"] = df_from_model[test_idxs].modelId.values.tolist()
    val_idxs = df_from_model["split"] == "val"
    splits["val"] = df_from_model[val_idxs].modelId.values.tolist()

    return splits

def store_primitive_parameters(
    size,
    shape,
    rotation,
    location,
    tapering,
    probability,
    color,
    filepath
):
    primitive_params = dict(
        size=size,
        shape=shape,
        rotation=rotation,
        location=location,
        tapering=tapering,
        probability=probability,
        color=color
    )

    fp = open(filepath, "wb")
    
    pickle.dump(
        primitive_params,
        fp
    )

    fp.close()


def load_primitive_parameters(
    filepath
):
    primitive_params = pickle.load(open(filepath, "rb"))
    return primitive_params

def find_files(path): return glob.glob(path)

def load_all_primitive_parameters(dir_path, prob_threshold, verbose=False):
    primitives = []
    i = 0
    # print("hererere!!!!")
    primitives_files = ["primitive_4.p",
    "primitive_10.p",
    "primitive_16.p",
    "primitive_17.p",
    "primitive_0.p",
    "primitive_3.p",
    "primitive_8.p",
    "primitive_7.p",
    "primitive_1.p",
    "primitive_12.p",
    "primitive_9.p",
    "primitive_14.p",
    "primitive_11.p",
    "primitive_5.p",
    "primitive_13.p",
    "primitive_2.p",
    "primitive_15.p",
    "primitive_6.p"]


    # for primitive_file in find_files(dir_path + "/*.p"):
    for primitive_file in primitives_files:
        print(dir_path, primitive_file)
        primitive_file = dir_path +  primitive_file
        print(primitive_file)
        _prim = load_primitive_parameters(primitive_file)
        # print(_prim)
        if _prim["probability"][0] >= prob_threshold:
            if verbose:
                print("Loading primitive nb %d.." % (i,))
            primitives.append(_prim)
        i += 1

    return primitives


def load_pose(file_path):
    trans = np.matrix([[ 1.,  1., -1.,  1.],
       [-1., -1.,  1., -1.],
       [-1., -1.,  1., -1.]])
    lines = []
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            lines.append(line.strip())
            line = fp.readline()

        box = np.matrix([[float(j) for j in i.split(" ")] for i in lines[1:9]])
        rt = np.matrix([[float(j) for j in i.split(" ")] for i in lines[10:13]])
        scale = float(lines[14])

    return box, rt, scale

def rotation_matrix_to_quaternion(m):
    # m_t = np.transpose(m)
    # i = np.matmul(m_t, m)
    # determinant = np.linalg.det(m)

    # qw = math.sqrt(1 + m[0,0] + m[1,1] + m[2,2]) /2
    # qx = (m[2,1] - m[1,2])/( 4 *qw)
    # qy = (m[0,2] - m[2,0])/( 4 *qw)
    # qz = (m[1,0] - m[0,1])/( 4 *qw)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m[0,0], m[0,1], m[0,2], m[1,0], m[1,1], m[1,2], m[2,0], m[2,1], m[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif ((m00 > m11) and (m00 > m22)): 
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S 
    elif (m11 > m22):
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2; 
        qw = (m02 - m20) / S;
        qx = (m01 + m10) / S; 
        qy = 0.25 * S;
        qz = (m12 + m21) / S; 
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2; 
        qw = (m10 - m01) / S;
        qx = (m02 + m20) / S;
        qy = (m12 + m21) / S;
        qz = 0.25 * S;

    return Quaternion([qw, qx, qy, qz])


def pose_to_string(pose):
    return json.dumps(list(list(list(x) for x in r) for r in pose))

def quat_pose_to_string(pose):
    return json.dumps(list(list(r) for r in pose))

def string_to_pose(string):
    return np.array(json.loads(string))