import os
import sys

from lib.models.direct_superquad_models import *
from lib.models.superquad_pose_models import *
from lib.models.direct_pose_models import *
from lib.models.loss_functions import *

def create_losses(loss_name):
    if loss_name == "L2":
        return l2_loss

def create_model(cfg):
    model_name = cfg.get('model_name', "direct_superquad")
    loss_name = cfg.get('loss_name', "L2")
    loss_func = create_losses(loss_name)

    if model_name == "direct_superquad":
        num_primitives =cfg.get('num_primitives', 11) 
        make_dense =cfg.get('make_dense', True) 
        model = DirectSuperQuadModel(num_primitives, make_dense)
    elif model_name == "superquad_pose":
        num_primitives =cfg.get('num_primitives', 11) 
        make_dense =cfg.get('make_dense', True) 
        model = SuperQuadPoseModel(num_primitives, make_dense)
    elif model_name == "direct_pose":
        make_dense =cfg.get('make_dense', True) 
        model = DirectPoseModel()

    train_model, eval_model = model.get_trainers()

    return model, train_model, eval_model, loss_func
