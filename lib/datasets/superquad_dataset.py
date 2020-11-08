import os
import sys
import pdb
sys.path.append(os.getcwd())

from PIL import Image
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import time
import random
# import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import glob
import quaternion as qua

from lib.utils.superquat_utils import *
from lib.zenlib.utils.proj_utils import *
from lib.zenlib.utils.transform_utils import *



class SuperQuadDataset(data.Dataset):
    def __init__(self, mode, dataset_info, rot_repr = "quat",  transforms=None, shuffle = True):
        print("********************** Loading SuperQuad Dataset **********************")
        self.rot_repr = rot_repr
        self.transforms = transforms
        self.dataset_info = dataset_info
        self.root = dataset_info["data_root"]
        self.parse_superquad_dataset(dataset_info)

        split = dataset_info["train_test_split"] 
        np.random.seed(1) # train test split need to stablize
        if shuffle:
            np.random.shuffle(self.data_list)
        
        if mode == 'train':
            split_num = np.floor(split * len(self.data_list)).astype(int)
            self.data_list = self.data_list[:split_num]
        elif mode == 'test':
            split_num = np.floor((1- split) * len(self.data_list)).astype(int)
            self.data_list = self.data_list[:split_num]

        
        self.length = len(self.data_list)
        self.M = len(self.primitives)

        
        print("Dataset Mode: ", mode)
        print("Dataset Length: ", len(self.data_list))
        print("********************** Finished SuperQuad Dataset **********************")

    def parse_superquad_dataset(self, dataset_info):
        
        self.primitives = load_all_primitive_parameters(dataset_info['quat_dir'], dataset_info["quat_threshold"])

        self.model_dict = read_pointxyz(os.path.join(self.root,'models'))
        self.models = []
        self.primitive_poses = []
        for p in self.primitives:
            p_r = Quaternion(p['rotation'])
            p_t = np.array(p['location'])[:,np.newaxis]
            self.primitive_poses.append((p_r, p_t))

        if dataset_info["all_classes"]:
            pass
        else:
            classes = dataset_info["classes"]
            all_images = []
            for class_name in classes:
                all_image = sorted(glob.glob(os.path.join(self.root, "data", class_name, "*-color.png")))
                all_image = [os.path.join(self.root, "data", class_name, i.split("/")[-1].split("-")[0])  for i in all_image]
                all_images = all_images + all_image
                self.models.append(self.model_dict[class_name])
        self.data_list = all_images
        

    def __getitem__(self, index):

        try:
            img = np.array(Image.open('{}-color.png'.format( self.data_list[index])))[:,:,:3]
            depth_info = np.load(('{}-depth.npz'.format(self.data_list[index])))
            # label = np.array(Image.open('{}-label.png'.format( self.data_list[index])))
            meta = np.load('{}-meta.npz'.format(self.data_list[index]))
        except Exception as e:
            print(e)
            print(self.data_list[index])

        model_id = meta['model_id']
        pose = meta['pose']
        pose = np.squeeze(pose)
        depth = depth_info['depth']
        
        # target_r = qua.as_float_array(qua.from_rotation_matrix(pose))
        target_r = mat_to_quat(pose[:3, :3]).elements
        target_t = np.squeeze(np.array([pose[:, 3:4].flatten()]))

        quad_rs = []
        quad_ts = []
        
        for p_r, p_t in self.primitive_poses:
            o_r = Quaternion(target_r)
            new_r = o_r * p_r.inverse
            new_t = np.matmul(o_r.rotation_matrix,  p_t)

            quad_rs.append(new_r.elements)
            quad_ts.append(new_t.squeeze())
            # quat_rs.append(p_r.elements)
            # quat_ts.append(p_t)

        quad_rs = np.hstack(quad_rs)
        quad_ts = np.hstack(quad_ts)


        img, depth, target_r, target_t, pose, quad_rs, quad_ts = img, \
               torch.from_numpy(depth.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.from_numpy(pose.astype(np.float32)), \
               torch.from_numpy(quad_rs.astype(np.float32)), \
               torch.from_numpy(quad_ts.astype(np.float32))

        if self.rot_repr == "quat":
            pass
        elif self.rot_repr == "6d":
            target_r = compute_orth6d_from_rotation_matrix(pose[:3, :3])
            quad_rs = compute_quat_to_orth6d(quad_rs.reshape(-1, 4))
            quad_rs = quad_rs.squeeze().view(quad_rs.shape[0], -1)
        else:
            raise Exception('Unkonwn Rotation Representation!!!')

        if self.transforms:
            img = self.transforms(img)
            
        return img, depth, target_r, target_t, pose, quad_rs, quad_ts 

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


if __name__ == "__main__":
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')

    dataset_info = {
        "dataset_name": "super_quat_dataset",
        "data_root": "/hdd/zen/data/Reallite/Rendering/take1/",
        "quat_dir": "temp_data/superquadric_primitives/chair1/",
        "quat_threshold": 0.5,
        "all_classes": False,
        "train_test_split": 0.7,
        "classes": ["357e2dd1512b96168e2b488ea5fa466a"]
    }
    batch_size = 8
    dataset = SuperQuadDataset("train", dataset_info, "6d")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=True, num_workers=8)
    for i, data in enumerate(loader):
        imgs, depth, target_r, target_t, pose, quad_rs, quad_ts  = data
        quad_rs = compute_orth_6d_to_quat(quad_rs.view(quad_rs.shape[0], -1, 6))
        quad_t = quad_ts[0].reshape(-1, 3).numpy()
        quad_r = quad_rs[0].reshape(-1, 4).numpy()
        
        img = imgs[0]
        display_primitives_pose_and_img(dataset.primitives, quad_r, quad_t, img)