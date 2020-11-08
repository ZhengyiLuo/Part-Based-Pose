
def add_datatype_parameters(parser):
    parser.add_argument(
        "--data_type",
        choices=[
            "image"
        ],
        default="image",
        help="The data type to be used (default=image)"
    )

    parser.add_argument(
        "--image_shape",
        type=lambda x: tuple(map(int, x.split(","))),
        default="3,224,224",
        help="The dimensionality of the image (default=(3,224,224)"
    )

def add_training_parameters(parser):
    """Add arguments to a parser that are related with the training of the
    network.
    """
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of times to iterate over the dataset (default=150)"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help=("Total number of steps (batches of samples) before declaring one"
              " epoch finished and starting the next epoch (default=500)")
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples in a batch (default=32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default 1e-4)"
    )
    parser.add_argument(
        "--lr_epochs",
        type=lambda x: list(map(int, x.split(","))),
        default="100,200,300",
        help="Training epochs with diminishing learning rate"
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=1.0,
        help=("Factor according to which the learning rate will be diminished"
              " (default=None)")
    )
    parser.add_argument(
        "--optimizer",
        choices=["Adam", "SGD"],
        default="Adam",
        help="The optimizer to be used (default=Adam)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help=("Parameter used to update momentum in case of SGD optimizer"
              " (default=0.9)")
    )


def add_dataset_parameters(parser):
    parser.add_argument(
        "--dataset_type",
        default="instance",
        choices=[
            "category",
            "instance"
        ],
        help="The type of the dataset type to be used"
    )

    parser.add_argument(
        "--model_tags",
        type=lambda x: x.split(":"),
        default=[],
        help="The tags to the model to be used for testing",
    )

def add_nn_parameters(parser):
    """Add arguments to control the design of the neural network architecture.
    """
    parser.add_argument(
        "--architecture",
        choices=["resnet18"],
        default="resnet18",
        help="Choose the architecture to train"
    )

    parser.add_argument(
        "--make_dense",
        action="store_true",
        help="When true use an additional FC before its regressor"
    )

def data_input_shape(args):
    if args.data_type == "image":
        return args.image_shape
