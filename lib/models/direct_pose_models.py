import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from lib.utils.eval_utils import *
from lib.models.pose_models import *

class DirectPoseModel(nn.Module):
    def __init__(self, rot_repr = "quat"):
        super(DirectPoseModel, self).__init__()
        input_channels = 512
        self.make_dense = True
        self._rot_repr = rot_repr

        self._features_extractor = models.resnet18(pretrained=True)
        self._features_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        self._features_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.obj_fc = nn.Conv3d(input_channels, input_channels, 1)
        self.obj_nonlin = nn.LeakyReLU(0.2, True)
        if self._rot_repr == "quat":            
            self._rotation_layer = nn.Conv3d(
                input_channels, 4, 1
            )
        elif self._rot_repr == "6d":
            self._rotation_layer = nn.Conv3d(
                input_channels, 6, 1
            )

       
    def forward(self, X):
        X = X.float() / 255.0
        x = self._features_extractor(X)
        x = self.obj_nonlin(self.obj_fc(x.view(-1, 512, 1, 1, 1)))
        if self._rot_repr == "quat":            
            obj_rots = self._rotation_layer(x).view(-1, 4)
            obj_rots = obj_rots / torch.norm(obj_rots, 2, -1, keepdim=True)
        elif self._rot_repr == "6d":
            obj_rots = self._rotation_layer(x).view(-1, 6)
        return  obj_rots

    def get_trainers(self):
        return direct_pose_train_model, direct_pose_eval_model

def direct_pose_train_model(dataset, model, loss_func, optimizer, device, batch_size = 128):
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
        pred_obj_rots = model(imgs)
        obj_loss= loss_func(pred_obj_rots, obj_rs)
        loss =obj_loss
        
        train_loss += loss
        total_num_sample += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        

    train_loss /= total_num_sample
    return loss_names, [train_loss]

def direct_pose_eval_model(dataset, model, loss_func, device, batch_size = 128):
    loss_names = ["loss", "avg_dist"]
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=True, num_workers=8)
    eval_loss = 0
    total_num_sample = 0
    M = dataset.M
    avg_distances = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, _, obj_rs, obj_ts, obj_poses, quat_rs, quat_ts, = data
            imgs, quat_rs, quat_ts, obj_rs = imgs.to(device), quat_rs.to(device), quat_ts.to(device), obj_rs.to(device)
            # Data inspection
            # vis_model_output(output, quat_rs, quat_ts, imgs, dataset.primitives)
            pred_obj_rots = model(imgs)
            obj_loss= loss_func(pred_obj_rots, obj_rs)
            loss =obj_loss

            dists = compare_rotation_distance(pred_obj_rots, obj_poses, dataset.models[0], dataset.rot_repr)
            avg_distances.append(np.array(dists).mean())

            eval_loss += loss  
            total_num_sample += 1

    eval_loss /= total_num_sample
    avg_distance = np.sum(avg_distances) / total_num_sample
    return loss_names, [eval_loss, avg_distance]