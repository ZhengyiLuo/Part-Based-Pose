#!/usr/bin/env python
import argparse

import json
import random
import os
import string
import sys

import numpy as np
import torch

from torch.utils.data import DataLoader

from arguments import add_datatype_parameters, add_nn_parameters, \
    add_dataset_parameters, add_training_parameters, data_input_shape
from output_logger import get_logger
from utils import parse_train_test_splits

from pose_estimation.common.dataset import get_dataset_type, \
    compose_transformations
from pose_estimation.common.model_factory import DatasetBuilder
from pose_estimation.common.batch_provider import BatchProvider
from pose_estimation.models import NetworkParameters, train_on_batch, \
    optimizer_factory
from pose_estimation.loss_functions import matrix_loss
from pose_estimation.datafactory import DataFactory

def moving_average(prev_val, new_val, b):
    return (prev_val*b + new_val) / (b+1)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))


def yield_infinite(iterable):
    while True:
        for item in iterable:
            yield item

def lr_schedule(optimizer, current_epoch, init_lr, factor, reductions):
    def inner(epoch):
        for i, e in enumerate(reductions):
            if epoch < e:
                return init_lr*factor**(-i)
        return init_lr*factor**(-len(reductions))

    for param_group in optimizer.param_groups:
        param_group['lr'] = inner(current_epoch)

    return optimizer

def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    git_head_hash = "foo"
    params["git-commit"] = git_head_hash
    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a network to predict primitives"
    )

    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )

    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )

    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trainined model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )

    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )

    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )

    parser.add_argument(
        "--cache_size",
        type=int,
        default=2000,
        help="The batch provider cache size"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )

    # Parse args
    add_nn_parameters(parser)
    add_dataset_parameters(parser)
    add_datatype_parameters(parser)
    add_training_parameters(parser)
    args = parser.parse_args(argv)

    if args.run_on_gpu: #and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Store the parameters for the current experiment in a json file
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in %s" %(experiment_tag, ))

    # Create two files to store the training and test evolution
    train_stats = os.path.join(experiment_directory, "train.txt")
    val_stats = os.path.join(experiment_directory, "val.txt")
    if args.weight_file is None:
        train_stats_f = open(train_stats, "w")
    else:
        train_stats_f = open(train_stats, "a+")
    train_stats_f.write((
        "epoch loss\n"
    ))

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))
    
    # TODO
    M = 11
    data_output_shape = (M, 7)

    # Create a factory that returns the appropriate data type based on the
    # input argument
    data_factory = DataFactory(
        args.data_type,
        tuple([data_input_shape(args), data_output_shape])
    )

    # Create a dataset instance to generate the samples for training
    training_dataset = get_dataset_type("matrix_loss")(
        (DatasetBuilder()
            .with_dataset(args.dataset_type)
            .build(args.dataset_directory)),
        data_factory,
        transform=compose_transformations(args.data_type)
    )

    training_loader = DataLoader(training_dataset, batch_size=32, num_workers=4,
                                pin_memory=True, drop_last=True, shuffle=True)

    # Build the model to be used for training
    network_params = NetworkParameters(args.architecture, M, False)
    model = network_params.network(network_params)

    # Move model to the device to be used
    model.to(device)
    
    # Check whether there is a weight file provided to continue training from
    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))
    model.train()

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(args, model)

    # Loop over the dataset multiple times
    losses = []
    for i in range(args.epochs):
        bar = get_logger(
            "matrix_loss",
            i+1,
            args.epochs,
            args.steps_per_epoch
        )
        
        j = 0
        for sample in training_loader:
            X, y_target = sample

            # if j == 0:
            #     import matplotlib.pyplot as plt
            #     import matplotlib.image as mpimg

                
            #     print(np.shape(X))
            #     print(X)
            #     img = X.numpy()[0]
            #     img = np.transpose(img, (1,2,0))
            #     img = img.reshape((224, 224, 3))
            #     print(img)

            #     imgplot = plt.imshow(img)
            #     print(imgplot)
            #     plt.show()

            # print(j)
            # j +=1
            # if j > 20:
            #     break
            # continue
            # print(X.shape)
            # print(y_target.shape)
            # #exit(1)
            
            
            # exit(1)

            X, y_target = X.to(device), y_target.to(device)

            # Train on batch
            batch_loss, metrics, debug_stats = train_on_batch(
                model,
                lr_schedule(
                    optimizer, i, args.lr, args.lr_factor, args.lr_epochs
                ),
                matrix_loss,
                X,
                y_target,
                device
            )

            # The losses
            bar.loss = moving_average(bar.loss, batch_loss, b)

            # Record in list
            losses.append(bar.loss)

            # TODO: Update the file that keeps track of the statistics
            if (j % 50) == 0:
                train_stats_f.write(
                    ("%d %5.8f") %(
                    i, bar.loss)
                )
                train_stats_f.write("\n")
                train_stats_f.flush()
            j += 1
            bar.next()

            if j >= args.steps_per_epoch:
                break

        # Finish the progress bar and save the model after every epoch
        bar.finish()

        if (i % 5) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    experiment_directory,
                    "model_%d" % (i + args.continue_from_epoch,)
                )
            )

    torch.save(model.state_dict(),
                os.path.join(experiment_directory, "model_final"))

    # TODO: print final training stats
    print([
        sum(losses[args.steps_per_epoch:]) / float(args.steps_per_epoch),
        sum(losses[:args.steps_per_epoch]) / float(args.steps_per_epoch)
    ])


if __name__ == "__main__":
    main(sys.argv[1:])
