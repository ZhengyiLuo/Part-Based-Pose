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

''' Code import '''
from lib.models.model_creater import create_model
from lib.datasets.superquad_dataset import *
from lib.utils.superquat_utils import *
from lib.models.loss_functions import *
from lib.zenlib.utils.torch_utils import *
from lib.zenlib.utils.logger import *
from lib.zenlib.utils.transform_utils import *


def vis_model_output(output, gt_rs, gt_ts, imgs, primitives):
    
    for idx in range(imgs.shape[0]):
        img = np.transpose(imgs[idx].cpu().numpy(), (1,2,0))
        quat_t = output[0][idx].detach().cpu().numpy().reshape(11, 3)
        quat_r = output[1][idx].detach().cpu().numpy().reshape(11, 4)
        gt_t = gt_ts[idx].detach().cpu().numpy().reshape(11, 3)
        gt_r = gt_rs[idx].detach().cpu().numpy().reshape(11, 4)
        display_primitives_pose_and_img(primitives, quat_r, quat_t, img)
        # display_primitives_pose_and_img(primitives, gt_r, gt_t, img)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Superquad networks "
    )
    
    parser.add_argument("--out_base_dir",
                        help="Save the output files in that directory", default="model_output")
    parser.add_argument("--save_prediction_as_mesh", action="store_true",
                        help="When true store prediction as a mesh")
    parser.add_argument("--prob_threshold", type=float,
                        default=0.5, help="Probability threshold")
    parser.add_argument("--run_on_gpu", action="store_true", help="Use GPU")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--iter", type=int, default=None)
    args = parser.parse_args()

    ''' Load Configuration '''
    cfg_name = args.cfg
    cfg = yaml.safe_load(open("cfgs/{}.yml".format(args.cfg), 'r'))
    lr = cfg.get("lr", 0.0011)
    batch_size = cfg.get("batch_size", 128)
    train_epoch = cfg.get("train_epoch", 600)
    train_epoch_lr = cfg.get("train_epoch_lr", 100)
    tb_dir = cfg.get("tb_dir", "tb")
    log_dir = cfg.get("log_dir", "log")
    model_dir = cfg.get("model_dir", "model")
    save_model_interval = cfg.get("save_model_interval", 20)

    ''' Config Runtime '''
    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using..", device)
    output_dir = os.path.join(args.out_base_dir, args.cfg)
    tb_dir = os.path.join(output_dir, tb_dir)
    log_dir = os.path.join(output_dir, log_dir)
    model_dir = os.path.join(output_dir, model_dir)

    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    if not os.path.isdir(model_dir): os.makedirs(model_dir)
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    if not os.path.isdir(tb_dir): os.makedirs(tb_dir)

    tb_logger = SummaryWriter(tb_dir) if args.mode == "train" else None
    logger = create_logger(os.path.join(log_dir, "log.txt"))

    ''' Dataset Loading '''
    transforms = T.Compose([
        # T.Scale(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    dataset_info = cfg.get('dataset', None)
    if dataset_info == None:
        print("Dataset does not exist")
        pdb.set_trace()
    if dataset_info["dataset_name"] == "super_quat_dataset":
        train_dataset = SuperQuadDataset("train", dataset_info, transforms=transforms)
        test_dataset = SuperQuadDataset("test", dataset_info, transforms=transforms)


    ''' Load Model '''
    model, train_model, eval_model, loss_func = create_model(cfg)
    if args.iter is not None:
        print("Loading Model iteration", args.iter)
        model.load_state_dict(torch.load(os.path.join(
            model_dir, "model_{}".format(args.iter))))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_scheduler(
        optimizer,
        policy="lambda",
        nepoch_fix=train_epoch_lr,
        nepoch=train_epoch,
    )

    if args.mode == "train":
        for epoch in range(train_epoch):
            t_s = time.time()
            loss_names, losses = train_model(train_dataset, model, loss_func, optimizer, device, batch_size = batch_size)
            dt = time.time() - t_s
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            
            losses_str = " ".join(["{}: {:.4f}".format(x, y) for x, y in zip(loss_names, losses)])
            logger.info("====> Epoch: {} Time: {:.2f} {} lr: {:.5f}".format(epoch, dt, losses_str, lr))

            for name, loss in zip(loss_names, losses): tb_logger.add_scalar("train_" + cfg_name + name, loss, epoch)

            if (epoch + 1) % save_model_interval == 0:
                model_save_path = os.path.join(model_dir, "model_{}".format(epoch + 1))
                torch.save(model.state_dict(), model_save_path)
                loss_names, losses = eval_model(test_dataset, model, loss_func, device, batch_size = batch_size)
                losses_str = " ".join(["{}: {:.4f}".format(x, y) for x, y in zip(loss_names, losses)])
                logger.info("******** Eval: {} Time: {:.2f} {} lr: {:.5f}".format(epoch, dt, losses_str, lr))
                for name, loss in zip(loss_names, losses): tb_logger.add_scalar("eval_{}_{}".format(cfg_name, name), loss, epoch)

    elif args.mode == "eval":
        res = eval_model(test_dataset, model, loss_func, device, batch_size = batch_size)
        print(res)
