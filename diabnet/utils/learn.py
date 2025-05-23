import time
from collections import Counter, defaultdict

import numpy as np
import torch

import diabnet.learn.state_keeper as state
from diabnet.datasets.patient_history import PatientHistoryDataset
from diabnet.utils.model import get_optimizer
from diabnet.utils.vocab import Vocabulary


def get_train_variables(args, model, fabric):
    """
    Given args, and whether or not resuming training, return
    relevant train variales.

    Returns:
        start_epoch:  Index of initial epoch
        epoch_stats: Dict summarizing epoch by epoch results
        state_keeper: Object responsibile for saving and restoring training state
        models: Dict of models
        optimizers: Dict of optimizers, one for each model
        tuning_key: Name of epoch_stats key to control learning rate by
        num_epoch_sans_improvement: Number of epochs since last dev improvment, as measured by tuning_key
    """
    start_epoch = 1
    args.lr = args.init_lr
    epoch_stats = init_metrics_dictionary()
    state_keeper = state.StateKeeper(args)

    # Setup optimizers

    optimizer = get_optimizer(model, args)
    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.init_lr,
            epochs=args.epochs,
            steps_per_epoch=args.max_batches_per_train_epoch,
        )
    else:
        scheduler = None

    model, optimizer = fabric.setup(model, optimizer)
    num_epoch_sans_improvement = 0

    tuning_key = "dev_{}".format(args.tuning_metric)

    return (
        start_epoch,
        epoch_stats,
        state_keeper,
        model,
        optimizer,
        tuning_key,
        num_epoch_sans_improvement,
        scheduler,
    )


def init_metrics_dictionary():
    """
    An helper function. Return empty metrics dict.
    """
    stats_dict = defaultdict(list)
    stats_dict["best_epoch"] = 0
    return stats_dict


def sample_weights(labels, positive_frac):
    """
    Calculates the weights used by WeightedRandomSampler for balancing the batches.
    """
    label_counts = Counter(labels)
    weight_per_label = {1: positive_frac, 0: 1 - positive_frac}
    label_weights = {
        label: weight_per_label[label] / count for label, count in label_counts.items()
    }
    weights = []
    for y in labels:
        weights.append(label_weights[y])
    return torch.tensor(weights)


def get_dataset_loader(args, data):
    """
    Given args, and dataset class returns torch.utils.data.DataLoader
    Train/Dev/Attribution set are balanced. Test set is untouched.
    """

    if data.split_group in ["train", "dev"]:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights(
                data.metadata.future_outcome.tolist(), args.positive_patient_fraction
            ),
            num_samples=len(data),
            replacement=(data.split_group == "train"),
        )

    else:
        sampler = torch.utils.data.sampler.SequentialSampler(data)

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler,
        batch_size=args.train_batch_size
        if data.split_group == "train"
        else args.eval_batch_size,
        drop_last=True,
    )
    if args.debug:
        data_loader = torch.utils.data.DataLoader(
            dataset=data,
            sampler=batch_sampler,
            pin_memory=True,
            num_workers=0,
            collate_fn=bin_collate,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=data,
            sampler=batch_sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            persistent_workers=True,
            prefetch_factor=5,
            collate_fn=bin_collate,
        )

    print(f"Dataloader for split {data.split_group} was generated")
    return data_loader


def bin_collate(batch):
    try:
        if len(batch[0]) == 0:
            return None
    except IndexError:
        return None
    batch = batch[0]
    token_keys = [k for k in batch[0].keys() if k.endswith("tokens")]
    for key in token_keys:
        max_code_dim = max([batch[i][key].shape[1] for i in range(len(batch))])
        for sample in batch:
            pad_shape = (
                (0, 0),
                (0, max_code_dim - sample[key].shape[1]),
            )  # only pad after on code_dim
            sample[key] = np.pad(sample[key], pad_width=pad_shape)
    return torch.utils.data.default_collate(batch)


def get_dataset(
    dataset_type: str, split: str, vocabulary: Vocabulary, args, fraction=None
):
    if dataset_type != "PatientHistory":
        print(
            f"Train dataset {dataset_type} is not recognized. Defaulting to PatientHistoryDataset"
        )
        return PatientHistoryDataset(args, split, vocabulary, fraction)
