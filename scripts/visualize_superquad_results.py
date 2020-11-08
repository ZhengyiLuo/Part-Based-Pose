import os
import sys
import pdb
sys.path.append(os.getcwd())
os.environ["KMP_WARNINGS"] = "FALSE" 

''' Libaray import '''
import pdb
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from pyquaternion import Quaternion
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn as nn
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import tensorflow as tf
from tensorflow.python.framework import tensor_util

''' Code import '''
from lib.models.model_creater import create_model
from lib.datasets.superquad_dataset import *
from lib.utils.superquat_utils import *
from lib.models.loss_functions import *
from lib.zenlib.utils.torch_utils import *
from lib.zenlib.utils.logger import *
from lib.zenlib.utils.transform_utils import *

def analyze_results(summary_res):
    for exp_title, results in summary_res.items():
        if len(results["eval_dists"]) > 0:
            print(exp_title, results["eval_dists"][-1], np.min(results["eval_dists"]))
        


if __name__ == "__main__":
    output_dir = "model_output"
    tb_dirs = glob.glob(os.path.join(output_dir, "*/tb"))
    summary_res = {}
    for tb_dir in tb_dirs:
        tb_events = glob.glob(os.path.join(tb_dir, "events*"))

        train_losses = []
        eval_losses = []
        eval_dists = []

        for tb_event in tb_events:
            for e in tf.train.summary_iterator(tb_event):
                for v in e.summary.value:
                    tag = v.tag
                    value = v.simple_value
                    if tag.endswith("loss") and "train" in tag:
                        train_losses.append(value)
                    elif tag.endswith("loss") and "eval" in tag:
                        eval_losses.append(value)
                    elif tag.endswith("dist"):
                        eval_dists.append(value)
        summary_res[tb_dir.split("/")[1]] = {
            "train_losses": train_losses, 
            "eval_losses": eval_losses, 
            "eval_dists": eval_dists
        }
    analyze_results(summary_res)