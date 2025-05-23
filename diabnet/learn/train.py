import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

from diabnet.learn.step import model_step
from diabnet.utils.eval import compute_eval_metrics
from diabnet.utils.learn import get_dataset_loader, get_train_variables, init_metrics_dictionary
from diabnet.utils.time_logger import TimeLogger

tqdm.monitor_interval = 0


def train_model(train_data, dev_data, model, args, fabric):
    """
    Train model and tune on dev set using args.tuning_metric. If model doesn't improve dev performance within
    args.patience epochs, then update the learning rate schedule (such as early stopping or halve the learning
    rate, and restore the model to the saved best and continue training. At the end of training, the function
    will restore the model to best dev version.

    Returns:
        epoch_stats: a dictionary of epoch level metrics for train and test
        returns models : dict of models, containing best performing model setting from this call to train
    """

    logger_epoch = (
        TimeLogger(args, 1, hierachy=2) if args.time_logger_verbose >= 2 else TimeLogger(args, 0)
    )
    (
        start_epoch,
        epoch_stats,
        state_keeper,
        model,
        optimizer,
        tuning_key,
        num_epoch_sans_improvement,
        scheduler,
    ) = get_train_variables(args, model, fabric)

    train_data_loader = get_dataset_loader(args, train_data)
    dev_data_loader = get_dataset_loader(args, dev_data)
    train_data_loader = fabric.setup_dataloaders(train_data_loader)
    dev_data_loader = fabric.setup_dataloaders(dev_data_loader)

    logger_epoch.log("Get train and dev dataset loaders")

    for epoch in range(start_epoch, args.epochs + 1):
        print("-------------\nEpoch {}:".format(epoch))

        for mode, data_loader in [
            ("Train", train_data_loader),
            ("Dev", dev_data_loader),
        ]:
            if_train = mode == "Train"
            key_prefix = mode.lower()
            (
                loss,
                golds,
                patient_golds,
                preds,
                probs,
                exams,
                pids,
                censor_times,
                days_to_final_censors,
                dates,
            ) = run_epoch(
                data_loader,
                train=if_train,
                truncate_epoch=True,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                fabric=fabric,
            )

            logger_epoch.log("Run epoch ({})".format(key_prefix))

            log_statement, epoch_stats, _ = compute_eval_metrics(
                args,
                loss,
                golds,
                patient_golds,
                probs,
                exams,
                pids,
                dates,
                censor_times,
                days_to_final_censors,
                epoch_stats,
                key_prefix,
            )
            logger_epoch.log("Compute eval metrics ({})".format(key_prefix))
            print(log_statement)

        # Save model if beats best dev (min loss or max c-index_{i,a})
        best_func, arg_best = (min, np.argmin) if "loss" in tuning_key else (max, np.argmax)
        improved = best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]
        if improved:
            num_epoch_sans_improvement = 0
            epoch_stats["best_epoch"] = arg_best(epoch_stats[tuning_key])
            state_keeper.save(model, optimizer, epoch, args.lr, epoch_stats)
            logger_epoch.log("Save improved model")
        else:
            num_epoch_sans_improvement += 1

        print(
            "---- Best Dev {} is {} at epoch {}".format(
                args.tuning_metric,
                epoch_stats[tuning_key][epoch_stats["best_epoch"]],
                epoch_stats["best_epoch"] + 1,
            )
        )
        if scheduler is not None:
            logger_epoch.log(f"Last Learning Rate in epoch:{scheduler.get_last_lr()}")

        elif num_epoch_sans_improvement >= args.patience:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0

            model, optimizer_state, _, _, _ = state_keeper.load()
            # Reset optimizers
            state_dict = optimizer_state
            optimizer = state_keeper.load_optimizer(optimizer, state_dict)
            # Reduce LR
            for param_group in optimizer.param_groups:
                param_group["lr"] *= args.lr_decay

            # Update lr also in args for resumable usage
            args.lr *= 0.5
        logger_epoch.log("Prepare for next epoch")
        logger_epoch.update()

    # Restore model to best dev performance, or last epoch when not tuning on dev
    model, _, _, _, _ = state_keeper.load()

    return epoch_stats, model


def run_epoch(data_loader, train, truncate_epoch, model, fabric, args, optimizer, scheduler=None):
    """
    Run model for one pass of data_loader, and return epoch statistics.
    Args:
        data_loader: Pytorch dataloader over some dataset.
        train: True to train the model and run the optimizers
        models: dict of models, where 'model' is the main model, and others can be critics, or meta-models
        truncate_epoch: used when dataset is too large, manually stop epoch after max_iteration without
                        necessarily spaning through the entire dataset.
        optimizers: dict of optimizers, one for each model
        args: general runtime args defined in by argparse

    Returns:
        avg_loss: epoch loss
        golds: labels for all samples in data_loader
        preds: model predictions for all samples in data_loader
        probs: model softmaxes for all samples in data_loader
        exams: exam ids for samples if available, used to cluster samples for evaluation.
    """
    data_iter = data_loader.__iter__()
    preds = []
    probs = []
    censor_times = []
    days_to_final_censors = []
    dates = []
    golds = []
    patient_golds = []
    losses = []
    exams = []
    pids = []
    logger = (
        TimeLogger(args, args.time_logger_step)
        if args.time_logger_verbose >= 3
        else TimeLogger(args, 0)
    )

    torch.set_grad_enabled(train)
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    batch_loss = 0
    num_batches_per_epoch = len(data_loader)

    if truncate_epoch:
        max_batches = args.max_batches_per_train_epoch if train else args.max_batches_per_dev_epoch
        num_batches_per_epoch = min(len(data_loader), (max_batches))
        logger.log("Truncate epoch @ batches: {}".format(num_batches_per_epoch))

    i = 0
    tqdm_bar = tqdm(data_iter, total=num_batches_per_epoch, leave=False)
    for batch in data_iter:
        if batch is None:
            warnings.warn("Empty batch")
            continue

        batch = prepare_batch(batch, args, model.inputs)
        logger.newline()
        logger.log("prepare data")

        step_results = model_step(batch, model, train, args, fabric)

        (
            loss,
            batch_preds,
            batch_probs,
            batch_golds,
            batch_patient_golds,
            batch_exams,
            batch_pids,
            batch_censors,
            batch_days_to_censor,
            batch_dates,
        ) = step_results
        batch_loss += loss
        logger.log("model step")

        if train:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        logger.log("model update")
        losses.append(batch_loss)
        batch_loss = 0

        preds.extend(batch_preds)
        probs.extend(batch_probs)
        golds.extend(batch_golds)
        patient_golds.extend(batch_patient_golds)
        dates.extend(batch_dates)
        censor_times.extend(batch_censors)
        days_to_final_censors.extend(batch_days_to_censor)
        exams.extend(batch_exams)
        pids.extend(batch_pids)
        logger.log("saving results")

        i += 1
        if i % int(0.1 * num_batches_per_epoch) == 0:
            tqdm_bar.update(int(0.1 * num_batches_per_epoch))

        if i >= num_batches_per_epoch:
            tqdm_bar.close()
            break

        logger.update()

    avg_loss = np.mean(losses)

    return (
        avg_loss,
        golds,
        patient_golds,
        preds,
        probs,
        exams,
        pids,
        censor_times,
        days_to_final_censors,
        dates,
    )


def prepare_batch(batch, args, model_inputs):
    keys_of_interest = [
        "y",
        "y_seq",
        "y_mask",
        "age",
        "time_at_event",
        "future_outcome",
        "days_to_censor",
        "patient_id",
    ]

    for key in batch.keys():
        if key in keys_of_interest or key in model_inputs:
            batch[key] = batch[key]
    return batch


def eval_model(eval_data, name, model, args, fabric):
    """
    Run model on test data, and return test stats (includes loss accuracy, etc)
    """
    logger_eval = (
        TimeLogger(args, 1, hierachy=2) if args.time_logger_verbose >= 2 else TimeLogger(args, 0)
    )
    logger_eval.log("Evaluating model")
    eval_stats = init_metrics_dictionary()
    logger_eval.log("Load model")

    data_loader = get_dataset_loader(args, eval_data)
    data_loader = fabric.setup_dataloaders(data_loader)
    logger_eval.log("Load eval data")

    (
        loss,
        golds,
        patient_golds,
        preds,
        probs,
        exams,
        pids,
        censor_times,
        days_to_final_censors,
        dates,
    ) = run_epoch(
        data_loader,
        train=False,
        truncate_epoch=(eval_data.split_group != "test"),
        fabric=fabric,
        model=model,
        optimizer=None,
        args=args,
    )
    logger_eval.log("Run eval epoch")

    log_statement, eval_stats, eval_preds = compute_eval_metrics(
        args,
        loss,
        golds,
        patient_golds,
        probs,
        exams,
        pids,
        dates,
        censor_times,
        days_to_final_censors,
        eval_stats,
        name,
    )
    print(log_statement)
    logger_eval.log("Compute eval metrics")
    logger_eval.update()

    return eval_stats, eval_preds
