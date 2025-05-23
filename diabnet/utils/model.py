import torch
from torch import nn


def get_optimizer(model, args):
    """
    Helper function to fetch optimizer based on args.
    """
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        return torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
        )
    else:
        raise Exception(f"Optimizer {args.optimizer} not supported!")
