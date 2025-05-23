import json
import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from os.path import dirname, realpath

import pandas as pd
import torch
from lightning.fabric import Fabric

sys.path.insert(0, dirname(dirname(realpath(__file__))))
import diabnet.learn.train as train
from diabnet.datasets.region_dataset import RegionalPatientDataset
from diabnet.models.factory import get_risk_model
from diabnet.utils.parsing import parse_args
from diabnet.utils.time_logger import TimeLogger
from diabnet.utils.vocab import PAD_CODE, Vocabulary

REGIONCODE_TO_NAME = {
    1084: "Capital Region Of Denmark",
    1085: "Region Zealand",
    1083: "Region of Southern Denmark",
    1082: "Central Denmark Region",
    1081: "North Denmark Region",
}


def parse_args():
    parser = ArgumentParser("Launch hold-region-out cross validation")
    parser.add_argument(
        "--configuration",
        required=True,
        help="Path to an args.json file with the args required to train and evaluate a model",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Path to the base directory for evaluating the model",
    )

    # Overwrite arguments that are not requried here
    args = parser.parse_args()
    config = Namespace(**json.load(open(args.configuration, "r")))
    config.train = True
    config.dev = True
    config.test = True
    config.attribution = False
    config.optimizer_state = None
    config.current_epoch = None
    config.lr = None
    config.epoch_stats = None
    config.step_indx = 1
    config.save_dir = args.save_dir

    return config


def run_crossvalidation_round(args):
    os.makedirs(args.experiment_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.experiment_dir, "args.json")):
        if not args.resume_experiment:
            print(
                "Trying to run already existing experiment, but have not specified --resume_experiment. Exiting to avoid overwriting"
            )
            sys.exit(1)
    else:
        args_dict = vars(args).copy()
        json.dump(args_dict, open(os.path.join(args.experiment_dir, "args.json"), "w"))

    torch.set_float32_matmul_precision("medium")
    fabric = Fabric(accelerator="auto", devices="auto")
    fabric.launch()

    logger_main = (
        TimeLogger(args, 1, hierachy=5, model_name=args.experiment_dir)
        if args.time_logger_verbose >= 1
        else TimeLogger(args, 0, model_name=args.experiment_dir)
    )
    logger_main.log("Now main.py starts...")

    vocabulary = Vocabulary.load_from_duckdb(
        args.duckdb_database, args.diag_level, args.atc_level, args.ydelse_level
    )

    print("CUDA:", torch.cuda.is_available())
    print("Building model...")
    model = get_risk_model(
        args.model_name,
        args=args,
        vocab_sizes=vocabulary.get_sizes(),
        padding_idxs=vocabulary.code_to_all_indexes(PAD_CODE),
    )
    print(model)

    train_data = RegionalPatientDataset(args, "train", vocabulary)
    dev_data = RegionalPatientDataset(args, "dev", vocabulary)

    epoch_stats, model = train.train_model(train_data, dev_data, model, args, fabric)
    print("Save train/dev results to {}".format(args.experiment_dir))
    logger_main.log("TRAINING")
    pickle.dump(
        epoch_stats,
        open(os.path.join(args.experiment_dir, "training_stats.pkl"), "wb"),
    )
    del epoch_stats, train_data
    logger_main.log("Dump results")

    print("-------------\nDev")
    dev_stats, dev_preds = train.eval_model(dev_data, "dev", model, args, fabric)
    print("Save dev results to {}".format(args.experiment_dir))
    logger_main.log("VALIDATION")
    pickle.dump(dev_stats, open(os.path.join(args.experiment_dir, "dev_stats.pkl"), "wb"))
    dev_preds = pd.DataFrame.from_dict(dev_preds)
    dev_preds.to_parquet(os.path.join(args.experiment_dir, "dev_preds.parquet"))

    del dev_stats, dev_preds, dev_data
    logger_main.log("Dump results")

    print()
    print("-------------\nTest")
    test_data = RegionalPatientDataset(args, "test", vocabulary)
    test_stats, test_preds = train.eval_model(test_data, "test", model, args, fabric)
    print("Save test results to {}".format(args.experiment_dir))
    logger_main.log("TESTING")

    pickle.dump(test_stats, open(os.path.join(args.experiment_dir, "test_stats.pkl"), "wb"))
    test_preds = pd.DataFrame.from_dict(test_preds)
    test_preds.to_parquet(os.path.join(args.experiment_dir, "test_preds.parquet"))
    del test_stats, test_preds, test_data
    logger_main.log("Dump results")


def main():
    args = parse_args()
    for test_region in REGIONCODE_TO_NAME.keys():
        args.experiment_dir = f"{args.save_dir}/{test_region}"
        args.test_region = test_region
        run_crossvalidation_round(args)


if __name__ == "__main__":
    main()
