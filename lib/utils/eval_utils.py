import torch 
from lib.zenlib.utils.transform_utils import *
from lib.zenlib.utils.proj_utils import *

def compare_rotation_distance(pred_obj_rots, obj_poses, model_points, rot_repr = "quat"):
    if rot_repr == "quat":
        preds_rot = compute_rotation_matrix_from_quaternion(pred_obj_rots)
    elif rot_repr == "6d":
        preds_rot = compute_rotation_matrix_from_ortho6d(pred_obj_rots)
    dists = []
    for i in range(preds_rot.shape[0]):
        rot1 = preds_rot[i].detach().cpu().numpy()
        rot1_4 = np.eye(4)
        rot1_4[:3,:3] = rot1
        rot2 = np.eye(4)
        rot2[:3,:3] = obj_poses[i][:3,:3].detach().cpu().numpy()
        dist = compare_rotation(model_points, rot1_4[:3,:4], rot2[:3,:4])
        dists.append(dist)
    return dists