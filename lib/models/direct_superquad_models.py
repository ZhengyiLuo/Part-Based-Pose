import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from lib.models.pose_models import *

class DirectSuperQuadModel(nn.Module):
    def __init__(self, n_primitives, make_dense=False, rot_repr = "quat"):
        super(DirectSuperQuadModel, self).__init__()
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
        return self._primitive_layer(x.view(-1, 512, 1, 1, 1))
        
    def get_trainers(self):
        return direct_super_quad_train_model, direct_super_quad_eval_model

def direct_super_quad_train_model(dataset, model, loss_func, optimizer, device, batch_size = 128):
    loss_names = ["loss"]
    model.train()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=True, num_workers=8)
    train_loss = 0
    total_num_sample = 0
    M = dataset.M
    for i, data in enumerate(loader):
        imgs, _, obj_rs, obj_ts, obj_poses, quat_rs, quat_ts= data

        imgs, quat_rs, quat_ts = imgs.to(device), quat_rs.to(device), quat_ts.to(device)
        output = model(imgs)

        # Data inspection
        # vis_model_output(output, quat_rs, quat_ts, imgs, dataset.primitives)

        target_ys = torch.cat((quat_ts, quat_rs), dim = 1)
        output_ys = torch.cat((output[0], output[1]), dim = 1)
        loss = loss_func(output_ys, target_ys)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        # print(loss) 
        train_loss += loss
        total_num_sample += 1

    train_loss /= total_num_sample
    return loss_names, [train_loss]

def direct_super_quad_eval_model(dataset, model, loss_func, device, batch_size = 128):
    loss_names = ["loss"]
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=True, num_workers=8)
    eval_loss = 0
    total_num_sample = 0
    M = dataset.M
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, _, obj_rs, obj_ts, obj_poses, quat_rs, quat_ts = data
            imgs, quat_rs, quat_ts = imgs.to(device), quat_rs.to(device), quat_ts.to(device)
            output = model(imgs)

            target_ys = torch.cat((quat_ts, quat_rs), dim = 1)
            output_ys = torch.cat((output[0], output[1]), dim = 1)

            loss = loss_func(output_ys, target_ys)
            eval_loss += loss  
            total_num_sample += 1

    eval_loss /= total_num_sample
    return loss_names, [eval_loss]
