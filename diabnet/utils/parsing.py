import argparse
import hashlib
import json
import os
import re
import warnings

import torch

import diabnet.learn.state_keeper as state

POSS_VAL_NOT_LIST = "Flag {} has an invalid list of values: {}. Length of slist must be >=1"


def parse_args(args_str=None):
    parser = argparse.ArgumentParser(description="PancNet Classifier")
    # What main steps to execute
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether or not to train model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether or not to run model on test set",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Whether or not to run model on dev set",
    )
    parser.add_argument(
        "--attribute",
        action="store_true",
        default=False,
        help="Whether or not to run attribution analysis (interpretation). "
        "Attribution is performed on test set",
    )
    parser.add_argument(
        "--interaction_attribute",
        action="store_true",
        default=False,
        help="Whether or not to run attribution analysis (interpretation) with interaction. "
        "Attribution is performed on test set and requires to specify codes_interaction_partners",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Disables Multiprocessing dataloading to enable debugging within __getitem__ of dataset",
    )

    # Device specification
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enable the gpu computation. If enabled but no CUDA is found then keep using CPU.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.environ.get("CPUS", 16),
        help="num workers for each data loader [default: 16]",
    )

    # Dataset setup
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="PatientHistory",
        help="Name of dataset to use. Default: 'PatientHistory",
    )
    parser.add_argument(
        "--dev_dataset",
        type=str,
        default="PatientHistory",
        help="Name of dataset to use. Default: 'PatientHistory",
    )

    parser.add_argument(
        "--duckdb_database",
        type=str,
        default="/users/projects/diabnet/data/pc/diabnet.db",
        help="Path of DuckDB database",
    )

    parser.add_argument(
        "--modalities", nargs="+", default=["diag"], help="Which modalities to include"
    )
    parser.add_argument(
        "--month_endpoints",
        nargs="+",
        type=int,
        default=[3, 6, 12, 36],
        help="List of month endpoints at which to generate risk prediction.",
    )
    parser.add_argument(
        "--relative_endpoints",
        action="store_true",
        default=False,
        help="Move the endpoint realtive to exclusion (endpoint = endpoint + exclusion)",
    )
    parser.add_argument(
        "--pad_size",
        type=int,
        default=250,
        help="Padding the trajectories to how long for training. Default: Pad every trajectory to the max_events_length. For BinnedTrajectory this is the number of bins",
    )
    parser.add_argument(
        "--trajectory_lookback",
        type=int,
        default=10,
        help="The time length (in years) each subtrajectory has available. This is a maxmimum value, as not all trajectories has this available.",
    )
    parser.add_argument(
        "--single_index",
        action="store",
        nargs="+",
        default=None,
        help="Send all indexes to 1 single index for each modality (Except padding). Try to model frequency of visits instead of codes. Argument is one or more modalities",
    )
    parser.add_argument(
        "--exclude_codes",
        action="store",
        default=None,
        help="Path to a txt file that contains codes to exlcude. Format is a code per line. The full code should be supplied",
    )
    parser.add_argument(
        "--min_events_length",
        type=int,
        default=5,
        help="Min num of events to include a patient",
    )
    parser.add_argument(
        "--min_unique_events",
        type=int,
        default=1,
        help="Minimum number of distinct events (truncated to specified code level)",
    )
    parser.add_argument(
        "--min_years_history",
        type=int,
        default=0,
        help="Minimum number of valid event history required for each patient",
    )
    parser.add_argument(
        "--min_events_diag",
        type=int,
        default=1,
        help="Min num of events to include a patient",
    )
    parser.add_argument(
        "--min_events_prescription",
        type=int,
        default=1,
        help="Min num of events to include a patient",
    )
    parser.add_argument(
        "--min_events_ydelse",
        type=int,
        default=1,
        help="Min num of events to include a patient",
    )
    parser.add_argument(
        "--sex",
        type=int,
        default=[1, 2],
        nargs="+",
        help="Run a model only on the chosen sex (1 or 2, default is both)",
        choices=[1, 2],
    )
    parser.add_argument(
        "--codes_required_in_trajectory",
        default=None,
        help="txt file with codes. If specified trajectories generated will include atleast one of the codes. Mostly to only create attributions for trajectories in a targeted manner.",
    )
    parser.add_argument(
        "--interaction_partners",
        default=None,
        help="txt file with codes. must be specified for interaction_attribute to work. These are the codes to search for interactions with.",
    )
    parser.add_argument(
        "--exclusion_interval",
        type=int,
        default=0,
        help="Exclude events before end of trajectory, default: 0 (month).",
    )
    parser.add_argument(
        "--positive_trajectory_fraction",
        type=float,
        default=0.5,
        help="Sampling probability for a positive patient to take a positive trajectory (only matters for training)",
    )
    parser.add_argument(
        "--positive_patient_fraction",
        type=float,
        default=0.5,
        help="Sampling probability for a positive patient (only matters for training)",
    )
    parser.add_argument(
        "--negatives_per_positive",
        type=int,
        default=1,
        help="Number of negatives to sample per positive. Only applies for MatchedDataset.",
    )
    parser.add_argument(
        "--max_eval_indices",
        type=int,
        default=250,
        help="Max number of trajectories to include for each patient during dev and test. ",
    )
    parser.add_argument(
        "--max_train_indices",
        type=int,
        default=None,
        help="Max Number of trajectories to include for each patient during training. (default = number of month endpoints) ",
    )
    parser.add_argument(
        "--min_age_assessment",
        type=int,
        default=0,
        help="Minimum age to consider for risk assessment",
    )
    parser.add_argument(
        "--min_year_admission",
        type=int,
        default=1995,
        help="Minimum year for risk assessment",
    )
    parser.add_argument(
        "--max_year_admission",
        type=int,
        default=2022,
        help="Max year for risk assessment",
    )
    parser.add_argument(
        "--negative_lookahead",
        type=float,
        default=2.0,
        help="Amount of time that should be available from last event in trajectory to censoring time (In case of censoring with disease)",
    )

    parser.add_argument(
        "--patients_exclusion_file",
        type=str,
        default=None,
        help="Path to a file with patients to remove from training. Format is a csv that has a column named pid.",
    )
    parser.add_argument(
        "--patients_inclusion_file",
        type=str,
        default=None,
        help="Path to a file with patients to include during training. Format is a csv that has a column named pid.",
    )
    # Hyper-params for model training
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers to use for sequential NNs.",
    )
    parser.add_argument(
        "--fusion_layers",
        type=int,
        default=1,
        help="Number of layers to use in the fusion module",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Number of heads to use for multihead attention. Only relevant for transformer.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Representation size at end of network.",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="EmbeddingLayer",
        help='Embedding type for codes. Choose from ["EmbeddingLayer","OneHot"]',
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="Transformer",
        help='Encoder to use. Choose from ["Transformer","GRU","LSTM", "MLP"]',
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MultiTrajectory",
        help="Model Architecture",
    )
    parser.add_argument(
        "--pooler",
        type=str,
        default="GlobalAverage",
        help='Pooling mechanism. Choose from ["Attention","GlobalAverage","GlobalMax"]',
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="CumulativeProbability",
        help='Classifier head. Choose from ["CumulativeProbability","Linear"]',
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout value for the neural network model.",
    )
    parser.add_argument(
        "--use_time_embed",
        action="store_true",
        default=False,
        help="Whether or not to condition embeddings by their relative time to the outcome date.",
    )
    parser.add_argument(
        "--use_age_embed",
        action="store_true",
        default=False,
        help="Whether or not to condition embeddings by the age at administration.",
    )
    parser.add_argument(
        "--time_embed_dim",
        type=int,
        default=128,
        help="Representation layer size for time embeddings.",
    )

    # Learning Hyper-params
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="binary_cross_entropy_with_logits",
        help="loss function to use, available: [Xent (default), MSE]",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="The optimizer to use during training. Choose from [default: adam, adagrad, sgd]",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="Batch size used when training the model.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size used when evaluating the model. Note that evaluation step takes all valid "
        "partial trajectories from each patient, therefore would consume higher memory per batch. "
        "One can adjust this accordingly using this option. ",
    )
    parser.add_argument(
        "--max_batches_per_train_epoch",
        type=int,
        default=10000,
        help="max batches to per train epoch. [default: 10000]",
    )
    parser.add_argument(
        "--max_batches_per_dev_epoch",
        type=int,
        default=10000,
        help="max batches to per dev epoch. [default: 10000]",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        default=0.001,
        help="The initial learning rate [default: 0.001]",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=1.0,
        help="Decay of learning rate [default: no decay (1.)]",
    )
    parser.add_argument(
        "--schedule_lr",
        action="store_true",
        default=False,
        help="Use a OneCycleLR scheduler",
    )
    parser.add_argument("--momentum", type=float, default=0, help="Momentum to use with SGD")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="L2 Regularization penaty [default: 0]",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs without improvement on dev before halving learning rate or early "
        "stopping. [default: 5]",
    )
    parser.add_argument(
        "--tuning_metric",
        type=str,
        default="36month_auroc",
        help="Metric to judge dev set results. Possible options include auc, loss, accuracy and etc.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Total number of epochs for training [default: 20].",
    )

    # evaluation
    parser.add_argument(
        "--eval_auroc",
        action="store_true",
        default=False,
        help="Whether to calculate AUROC",
    )
    parser.add_argument(
        "--eval_auprc",
        action="store_true",
        default=False,
        help="Whether to calculate AUPRC",
    )
    parser.add_argument(
        "--eval_mcc",
        action="store_true",
        default=False,
        help="Whether to calculate MCC",
    )
    parser.add_argument(
        "--eval_c_index",
        action="store_true",
        default=False,
        help="Whether to calculate c-Index",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="The output directory for the experiment.",
    )
    parser.add_argument(
        "--exp_id", type=str, default="debug", help="The identifier/name for each run"
    )
    parser.add_argument(
        "--time_logger_verbose",
        type=int,
        default=2,
        help="Verbose of logging (1: each main, 2: each epoch, 3: each step). Default: 2.",
    )
    parser.add_argument(
        "--time_logger_step",
        type=int,
        default=1,
        help="Log the time elapse every how many iterations - 0 for no logging.",
    )
    parser.add_argument(
        "--resume_experiment",
        action="store_true",
        help="Whether ot not to load from the specified directory. Only implemented for model evaluation",
    )

    args = parser.parse_args() if args_str is None else parser.parse_args(args_str.split())

    # Generate a few more flags to help model construction
    args.experiment_dir = os.path.join(args.save_dir, args.exp_id)
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = "cuda" if args.cuda else "cpu"

    # Resume experiments
    if args.resume_experiment:
        overwrite_args = json.load(open(os.path.join(args.experiment_dir, "args.json"), "r"))
        for key, value in overwrite_args.items():
            if key not in [
                "train",
                "dev",
                "test",
                "attribute",
                "debug",
                "resume_experiment",
                "eval_batch_size",
                "max_eval_indices",
            ]:
                setattr(args, key, value)
        args.snapshot = state.get_model_path(args.experiment_dir)
        args.device = args.device if torch.cuda.is_available() else "cpu"

    args.num_years = max(args.month_endpoints) / 12

    # Check whether the current args is legal.
    if args.train:
        assert args.dev, Exception(
            "[E] --dev is disabled. The dev dataset is required if --train for tuning purpose."
        )

    if args.test or args.dev:
        if (
            not any(
                [
                    True
                    for argument, values in args.__dict__.items()
                    for metric in argument.split("_")[-1:]
                    if metric in args.tuning_metric and values
                ]
            )
            and args.tuning_metric != "loss"
        ):
            raise Exception(
                "[E] Tuning metric {} is not computed in Eval metric! Aborting.".format(
                    args.tuning_metric
                )
            )

        assert any([args.eval_auroc, args.eval_auprc, args.eval_mcc, args.eval_c_index]), (
            Exception(
                "[E] At least one evaluation metric needs to be enabled. "
                "Choose one or more from AUPRC, AUROC, MCC or c-Index"
            )
        )

    if args.num_heads is not None and args.encoder != "Transformer":
        raise Exception(
            "[W] The `num_heads` is intended to work with Transformer only. "
            "Setting this for `{}` will have no effects.".format(args.encoder)
        )
    elif args.encoder == "Transformer":
        args.num_heads = args.num_heads if args.num_heads else 16
    if args.test or args.dev:
        if (
            not any(
                [
                    True
                    for argument, values in args.__dict__.items()
                    for metric in argument.split("_")[-1:]
                    if metric in args.tuning_metric and values
                ]
            )
            and args.tuning_metric != "loss"
        ):
            raise Exception(
                "[E] Tuning metric {} is not computed in Eval metric! Aborting.".format(
                    args.tuning_metric
                )
            )

        assert any([args.eval_auroc, args.eval_auprc, args.eval_mcc, args.eval_c_index]), (
            Exception(
                "[E] At least one evaluation metric needs to be enabled. "
                "Choose one or more from AUPRC, AUROC, MCC or c-Index"
            )
        )

    if args.num_heads is not None and args.encoder != "Transformer":
        raise Exception(
            "[W] The `num_heads` is intended to work with transformer only. "
            "Setting this for `{}` will have no effects.".format(args.encoder)
        )
    if args.relative_endpoints and not args.resume_experiment:
        args.month_endpoints = [m + args.exclusion_interval for m in args.month_endpoints]

    eval_month = re.search(r"^\d+", args.tuning_metric)
    if not eval_month:
        raise Exception(
            "[E] Selected evaluation month was not recognized. Try to specify in format 36month_auroc"
        )
    elif int(eval_month.group()) not in args.month_endpoints:
        warnings.warn(
            f"The selected eval month was not found in month_endpoints. Defaulting to last available endpoint: {args.month_endpoints[-1]} months"
        )
        args.tuning_metric = re.sub(
            pattern=r"^\d+",
            repl=str(args.month_endpoints[-1]),
            string=args.tuning_metric,
        )
    if args.interaction_attribute and not args.interaction_partners:
        raise TypeError(
            "codes_interaction_partners must be specified for interaction_attribute to work"
        )
    # Set up initial state for learning rate
    args.optimizer_state = None
    args.current_epoch = None
    args.lr = None
    args.epoch_stats = None
    args.step_indx = 1

    return args


def parse_dispatcher_config(config):
    """
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.

    Args:
        config - experiment_config json file
    Returns:
        jobs - a list of flag strings, each of which encapsulates one job.
        * Example: --train --cuda --dropout=0.1 ...

    """
    jobs = [""]
    hyperparameter_space = config["search_space"]
    hyperparameter_space_flags = hyperparameter_space.keys()
    hyperparameter_space_flags = sorted(hyperparameter_space_flags)
    for ind, flag in enumerate(hyperparameter_space_flags):
        possible_values = hyperparameter_space[flag]

        children = []
        if len(possible_values) == 0 or type(possible_values) is not list:
            raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
        for value in possible_values:
            for parent_job in jobs:
                if type(value) is bool:
                    if value:
                        new_job_str = "{} --{}".format(parent_job, flag)
                    else:
                        new_job_str = parent_job
                elif type(value) is list:
                    val_list_str = " ".join([str(v) for v in value])
                    new_job_str = "{} --{} {}".format(parent_job, flag, val_list_str)
                elif value is None:
                    new_job_str = parent_job
                else:
                    new_job_str = "{} --{} {}".format(parent_job, flag, value)
                children.append(new_job_str)
        jobs = children

    return jobs


class Dict2Args(object):
    """
    A helper class for easier attribution retrieval for dict.
    """

    def __init__(self, d):
        self.__dict__ = d


class Yaml2Args(Dict2Args):
    """
    A helper class for easier attribution retrieval for an YAML input.
    """

    def __init__(self, d):
        for item in d:
            if len(d[item]) == 1:
                d[item] = d[item][0]

        super(Yaml2Args, self).__init__(d)


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()
