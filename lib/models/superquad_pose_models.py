import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from lib.models.pose_models import *
from lib.utils.eval_utils import *

class SuperQuadPoseModel(nn.Module):
    def __init__(self, n_primitives, make_dense=False, rot_repr = "quat"):
        super(SuperQuadPoseModel, self).__init__()
        self._rot_repr = rot_repr
        self.n_primitives = n_primitives
        self.make_dense = True

        self._features_extractor = models.resnet18(pretrained=True)
        self._features_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        self._features_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._primitive_layer = self.primitive_layer(
            self.n_primitives,
            512
        )

        self.obj_pose_layer = nn.Sequential(
            nn.Linear(77, 256), nn.Sigmoid(),
            nn.Linear(256, 4)
        )

    def primitive_layer(self, n_primitives, input_channels):
        pose_module = Pose(n_primitives, self._build_modules(n_primitives, input_channels))
        return pose_module

    def _build_modules(self, n_primitives, input_channels):
        modules = {
            "translations": Prim_Translation(n_primitives, input_channels, self.make_dense),
            "rotations": Prim_Rotation(n_primitives, input_channels, self.make_dense, self._rot_repr),
        }
        return modules
       
    def forward(self, X):
        X = X.float() / 255.0
        x = self._features_extractor(X)
        translations, rotations = self._primitive_layer(x.view(-1, 512, 1, 1, 1))
        quad_poses = torch.cat((translations, rotations), dim = 1)
        obj_rots = self.obj_pose_layer(quad_poses)
        return quad_poses, obj_rots

    def get_trainers(self):
        return superquad_pose_train_model, superquad_pose_eval_model

def superquad_pose_train_model(dataset, model, loss_func, optimizer, device, batch_size = 128):
    loss_names = ["loss"]
    model.train()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=True, num_workers=8)
    train_loss = 0
    total_num_sample = 0
    M = dataset.M
    for i, data in enumerate(loader):
        imgs, _, obj_rs, obj_ts, obj_poses, quat_rs, quat_ts = data
        imgs, quat_rs, quat_ts, obj_rs = imgs.to(device), quat_rs.to(device), quat_ts.to(device), obj_rs.to(device)
        # Data inspection
        # vis_model_output(output, quat_rs, quat_ts, imgs, dataset.primitives)
        pred_quad_poses, pred_obj_rots = model(imgs)
        target_quads = torch.cat((quat_ts, quat_rs), dim = 1)
        quad_loss = loss_func(pred_quad_poses, target_quads)
        obj_loss= loss_func(pred_obj_rots, obj_rs)
        loss = quad_loss + obj_loss
        
        train_loss += loss
        total_num_sample += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        

    train_loss /= total_num_sample
    return loss_names, [train_loss]

def superquad_pose_eval_model(dataset, model, loss_func, device, batch_size = 128):
    loss_names = ["loss", "avg_dist"]
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=True, num_workers=8)
    eval_loss = 0
    total_num_sample = 0
    M = dataset.M
    avg_distances = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, _, obj_rs, obj_ts, obj_poses, quat_rs, quat_ts= data
            imgs, quat_rs, quat_ts, obj_rs = imgs.to(device), quat_rs.to(device), quat_ts.to(device), obj_rs.to(device)
            # Data inspection
            # vis_model_output(output, quat_rs, quat_ts, imgs, dataset.primitives)
            pred_quad_poses, pred_obj_rots = model(imgs)
            target_quads = torch.cat((quat_ts, quat_rs), dim = 1)
            quad_loss = loss_func(pred_quad_poses, target_quads)
            obj_loss= loss_func(pred_obj_rots, obj_rs)
            loss = quad_loss + obj_loss

            dists = compare_rotation_distance(pred_obj_rots, obj_poses, dataset.models[0], dataset.rot_repr)
            avg_distances.append(np.array(dists).mean())

            eval_loss += loss  
            total_num_sample += 1

    eval_loss /= total_num_sample
    avg_distance = np.sum(avg_distances) / total_num_sample
    return loss_names, [eval_loss, avg_distance]