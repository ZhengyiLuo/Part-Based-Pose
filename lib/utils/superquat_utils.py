import numpy as np
import pickle
import glob
import math
from pyquaternion import Quaternion
import os

from functools import reduce
import numpy as np
import pickle
import trimesh
from pyquaternion import Quaternion
import math
import matplotlib.pyplot as plt

def vis_model_output(output, gt_rs, gt_ts, imgs, primitives):
    
    for idx in range(imgs.shape[0]):
        img = np.transpose(imgs[idx].cpu().numpy(), (1,2,0))
        quat_t = output[0][idx].detach().cpu().numpy().reshape(11, 3)
        quat_r = output[1][idx].detach().cpu().numpy().reshape(11, 4)
        gt_t = gt_ts[idx].detach().cpu().numpy().reshape(11, 3)
        gt_r = gt_rs[idx].detach().cpu().numpy().reshape(11, 4)
        display_primitives_pose_and_img(primitives, quat_r, quat_t, img)
        # display_primitives_pose_and_img(primitives, gt_r, gt_t, img)


def display_primitives_pose_and_img(primitives, quat_r, quat_t, img):
    primitive_list = []
    for i, p in enumerate(primitives):
        x_tr, y_tr, z_tr, prim_pts =\
            points_on_sq_surface(
                p["size"][0],
                p["size"][1],
                p["size"][2],
                p["shape"][0],
                p["shape"][1],
                Quaternion(quat_r[i]).rotation_matrix.reshape(3, 3),
                np.array(quat_t[i]).reshape(3, 1),
                p["tapering"][0],
                p["tapering"][1],
                None,
                None, 
                n_samples = 30
            )
        primitive_list.append((prim_pts, p['color']))
    display_primitives(primitive_list, img)

def display_primitives(primitive_list, gt_img):
    fig = plt.figure(dpi = 120)
    ax = fig.add_subplot(131, projection='3d')
    transform_world_to_matplot = np.array([
        [1,0,0],
        [0,0,1],
        [0,1,0]
    ])
    for prim_pts, color in primitive_list:
        prim_pts = np.matmul(prim_pts.T, transform_world_to_matplot)
        ax.scatter(prim_pts[:,0], prim_pts[:,1], prim_pts[:,2], c =color)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.5, -0.5)
    ax.set_zlim(-0.5, 0.5)
    minbound = -0.5
    maxbound = 0.5
    
    ax = fig.add_subplot(132)
    ax.set_aspect('equal')
    for prim_pts, color in primitive_list:
        prim_pts = np.matmul(prim_pts.T, transform_world_to_matplot)
        ax.scatter(prim_pts[:,0], prim_pts[:,2], c =color)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)

    ax = fig.add_subplot(133)
    ax.imshow(gt_img)

    plt.show()
    
    

def rotation_matrix_to_quaternion(m):
    # m_t = np.transpose(m)
    # i = np.matmul(m_t, m)
    # determinant = np.linalg.det(m)

    qw = math.sqrt(1 + m[0,0] + m[1,1] + m[2,2]) /2
    qx = (m[2,1] - m[1,2])/( 4 *qw)
    qy = (m[0,2] - m[2,0])/( 4 *qw)
    qz = (m[1,0] - m[0,1])/( 4 *qw)
    return Quaternion([qw, qx, qy, qz])


def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)

def sq_surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)
    return x, y, z

def points_on_sq_surface(a1, a2, a3, e1, e2, R, t, Kx, Ky, R_obj = None, transform_matrix = None, n_samples=100):
    """Computes a SQ given a set of parameters and saves it into a np array
    """

    # scale_mat = np.identity(4)
    # scale = 10000
    # scale_mat[0,0] = scale
    # scale_mat[1,1] = scale
    # scale_mat[2,2] = scale
    # print(scale_mat)

    # new_transform = np.identity(4)
    # new_transform[:3, :4] = transform_matrix
    # print(new_transform)

    # tran = np.matmul(scale_mat, new_transform)
    # print(tran)

    # transform_matrix = tran[:3, :4]
    # print(transform_matrix)

    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    eta = np.linspace(-np.pi/2, np.pi/2, n_samples, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, n_samples, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, eta, omega)

    # Apply the deformations
    fx = Kx * z / a3
    fx += 1
    fy = Ky * z / a3
    fy += 1
    fz = 1

    x = x * fx
    y = y * fy
    z = z * fz

    # Get an array of size 3x10000 that contains the points of the SQ
    points_transformed = None
    points = None
    
    x_tr = None
    y_tr = None
    z_tr = None

    points = np.stack([x, y, z]).reshape(3, -1)
    # print("Init")
    # print(points)
    
    if transform_matrix is None:
        # print(R)
        # print(t)
        # print("--------------")
        # print(transform_matrix)
        #rotation_quat = rotation_matrix_to_quaternion(transform_matrix[:3,:3])
        # t = transform_matrix[:3, 3].reshape(3, 1)
        # R = R.rotation_matrix.reshape(3, 3)

        # print(t)

        # print(R)
        #exit(1)
        points_transformed = R.dot(points) + t
        #print(t)
        #points_transformed = points + t
        # print(points_transformed)

        # if R_obj is not None:
        #     points_transformed = R_obj.T.dot(points_transformed)

        x_tr = points_transformed[0].reshape(n_samples, n_samples)
        y_tr = points_transformed[1].reshape(n_samples, n_samples)
        z_tr = points_transformed[2].reshape(n_samples, n_samples)
    else:
        # T = np.identity(4)
        # T[:3, 3] = np.array([t[0, 0], t[1, 0], t[2, 0]])
        # # print("Translation")
        # # print(T)

        # r_obj = np.identity(4)
        # r_obj[:3, :3] = R_obj
        # # print("Robj")
        # # print(r_obj)

        # r = np.identity(4)
        # r[:3, :3] = R
        # print("Rot")
        # print(r)
        
        # transform_matrix = np.matmul(T, r.T)
        # transform_matrix = np.matmul(r_obj, transform_matrix)

        points = np.transpose(points)
        # print("Reshape")
        # print(points)

        ones = np.ones((np.shape(points)[0], 4))
        ones[:, :3] = points
        points_transformed = ones
        # print("Homogenized")
        # print(points_transformed)

        transform_4b4 = np.identity(4)
        transform_4b4[:3, :4] = transform_matrix
        #transform_4b4 = transform_matrix
        # print("Matrix")
        # print(transform_4b4)

        points_transformed = np.matmul(transform_4b4, points_transformed.T)
        # print("Transformed")
        # print(points_transformed)

        points_transformed = points_transformed[:3, :] # / points_transformed[:, 3:]
        # print("Unhomogenized")
        # print(points_transformed)

        x_tr = points_transformed[0].reshape(n_samples, n_samples)
        y_tr = points_transformed[1].reshape(n_samples, n_samples)
        z_tr = points_transformed[2].reshape(n_samples, n_samples)
    return x_tr, y_tr, z_tr, points_transformed

def superquadric_pose(R_obj, R, T):
    # print(R_obj)
    # print(R)
    # print(T)
    R_obj = Quaternion(R_obj).rotation_matrix.reshape(3, 3)
    R = Quaternion(R).rotation_matrix.reshape(3, 3)

    t = np.identity(4)
    t[:3, 3] = np.array([T[0], T[1], T[2]])
    # print("Translation")
    # print(t)

    r_obj = np.identity(4)
    r_obj[:3, :3] = R_obj
    # print("Robj")
    # print(r_obj)

    r = np.identity(4)
    r[:3, :3] = R
    # print("Rot")
    # print(r)

    transform = np.matmul(r_obj, np.matmul(t, r.T))

    return transform[:3, :4]

def points_on_cuboid(a1, a2, a3, e1, e2, R, t, n_samples=100):
    """Computes a cube given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    X = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1]
    ], dtype=np.float32)
    X[X == 1.0] = a1
    X[X == 0.0] = -a1

    Y = np.array([
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ], dtype=np.float32)
    Y[Y == 1.0] = a2
    Y[Y == 0.0] = -a2

    Z = np.array([
        [1, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1]
    ], dtype=np.float32)
    Z[Z == 1.0] = a3
    Z[Z == 0.0] = -a3

    points = np.stack([X, Y, Z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t

    assert points.shape == (3, 18)

    x_tr = points_transformed[0].reshape(2, 9)
    y_tr = points_transformed[1].reshape(2, 9)
    z_tr = points_transformed[2].reshape(2, 9)
    return x_tr, y_tr, z_tr, points_transformed


def _from_primitive_parms_to_mesh(primitive_params):
    if not isinstance(primitive_params, dict):
        raise Exception(
            "Expected dict and got {} as an input"
            .format(type(primitive_params))
        )
    # Extract the parameters of the primitives
    a1, a2, a3 = primitive_params["size"]
    e1, e2 = primitive_params["shape"]
    Kx, Ky = primitive_params["tapering"]
    t = np.array(primitive_params["location"]).reshape(3, 1)
    R = Quaternion(primitive_params["rotation"]).rotation_matrix.reshape(3, 3)
    
    R_obj = None
    # if "obj_pose" in primitive_params:
    #     R_obj = Quaternion(primitive_params["obj_pose"]).rotation_matrix.reshape(3, 3)
    
    full_transform = None
    # if "transform" in primitive_params:
    #     full_transform = np.array(primitive_params["transform"])

    # Sample points on the surface of its mesh
    _, _, _, V = points_on_sq_surface(a1, a2, a3, e1, e2, R, t, Kx, Ky, R_obj, full_transform)
    assert V.shape[0] == 3

    color = np.array(primitive_params["color"])
    color = (color*255).astype(np.uint8)

    # Build a mesh object using the vertices loaded before and get its
    # convex hull
    m = trimesh.Trimesh(vertices=V.T).convex_hull
    # Apply color
    for i in range(len(m.faces)):
        m.visual.face_colors[i] = color

    return m


def save_primitive_as_ply(primitive_params, filepath):
    m = _from_primitive_parms_to_mesh(primitive_params)
    # Make sure that the filepath endswith .obj
    if not filepath.endswith(".ply"):
        raise Exception(
            "The filepath should have an .ply suffix, instead we received {}"
            .format(filepath)
        )
    m.export(filepath, file_type="ply")

def save_params_as_ply(filepath, primitives):
    m = None
    i = 0
    for prim in primitives:
        _m = _from_primitive_parms_to_mesh(prim)
        m = trimesh.util.concatenate(_m, m)  
    m.export(filepath, file_type="ply")

def save_prediction_as_ply(primitive_files, filepath):
    if not isinstance(primitive_files, list):
        raise Exception(
            "Expected list and got {} as an input"
            .format(type(primitive_files))
        )
    m = None
    for p in primitive_files:
        # Parse the primitive parameters
        prim_params = pickle.load(open(p, "r"))
        _m = _from_primitive_parms_to_mesh(prim_params)
        m = trimesh.util.concatenate(_m, m)

    m.export(filepath, file_type="ply")

def get_colors(M):
    import seaborn as sns
    sns.set()
    return sns.color_palette("Paired")

import json

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


def find_files(path): return glob.glob(path)

def load_all_primitive_parameters(dir_path, prob_threshold, verbose=False):
    primitives = []
    i = 0
    primitives_files = [
    "primitive_4.p",
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

        primitive_file = os.path.join(dir_path, primitive_file)
        _prim = pickle.load(open(primitive_file, "rb"), encoding='latin1')
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

def mat_to_quat(m):
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