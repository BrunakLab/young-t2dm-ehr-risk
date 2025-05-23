import os
import pickle
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(realpath(__file__))))
import json

import pandas as pd
import torch
from lightning.fabric import Fabric

import diabnet.learn.attribute as attribute
import diabnet.learn.train as train
from diabnet.models.factory import get_risk_model
from diabnet.utils.learn import get_dataset
from diabnet.utils.parsing import parse_args
from diabnet.utils.time_logger import TimeLogger
from diabnet.utils.vocab import PAD_CODE, Vocabulary

if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args).copy()
    os.makedirs(args.experiment_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.experiment_dir, "args.json")):
        if not args.resume_experiment:
            print(
                "Trying to run already existing experiment, but have not specified --resume_experiment. Exiting to avoid overwriting"
            )
            sys.exit(1)
    else:
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

    vocabulary = Vocabulary.load_from_duckdb(args.duckdb_database)

    print("CUDA:", torch.cuda.is_available())

    if not args.resume_experiment:
        print("Building model...")
        model = get_risk_model(
            args.model_name,
            args=args,
            vocab_sizes=vocabulary.get_sizes(),
            padding_idxs=vocabulary.code_to_all_indexes(PAD_CODE),
        )
    else:
        print("Loading model...")
        model = torch.load(args.snapshot)
        model = model.to(args.device)

    print(model)
    print("Working threads: ", torch.get_num_threads())
    if torch.get_num_threads() < args.num_workers:
        torch.set_num_threads(args.num_workers)
        print("Adding threads count to {}.".format(torch.get_num_threads()))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in [
            "optimizer_state",
        ]:
            print("\t{}={}".format(attr.upper(), value))
    logger_main.log("Build model")

    print()
    if args.train:
        train_data = get_dataset(args.train_dataset, "train", vocabulary, args)
        dev_data = get_dataset(args.dev_dataset, "dev", vocabulary, args)
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
    if args.test:
        print("-------------\nTest")
        test_data = get_dataset("PatientHistory", "test", vocabulary, args)
        test_stats, test_preds = train.eval_model(test_data, "test", model, args, fabric)
        print("Save test results to {}".format(args.experiment_dir))
        logger_main.log("TESTING")

        pickle.dump(test_stats, open(os.path.join(args.experiment_dir, "test_stats.pkl"), "wb"))
        test_preds = pd.DataFrame.from_dict(test_preds)
        test_preds.to_parquet(os.path.join(args.experiment_dir, "test_preds.parquet"))
        del test_stats, test_preds, test_data
        logger_main.log("Dump results")

    print()
    if args.attribute:
        attribution_set = get_dataset("PatientHistory", "test", vocabulary, args, fraction=0.05)
        print("-------------\nAttribution")
        code_attribution, age_attribution = attribute.compute_attribution(
            attribution_set, model, vocabulary, args
        )
        print("Save attribution results to {}".format(args.experiment_dir))
        logger_main.log("ATTRIBUTION")
        code_attribution = pd.DataFrame.from_dict(code_attribution)
        code_attribution.to_parquet(os.path.join(args.experiment_dir, "code_attributions.parquet"))
        del code_attribution
        age_attribution = pd.DataFrame.from_dict(age_attribution)
        age_attribution.to_parquet(os.path.join(args.experiment_dir, "age_attributions.parquet"))
        logger_main.log("Dump results")

    if args.interaction_attribute:
        attribution_set = get_dataset("PatientHistory", "test", vocabulary, args, fraction=0)
        print("-------------\nInteraction Attribution")
        code_attribution = attribute.compute_interaction(attribution_set, model, vocabulary, args)
        print("Save attribution results to {}".format(args.experiment_dir))
        logger_main.log("ATTRIBUTION")
        code_attribution = pd.DataFrame.from_dict(code_attribution)
        code_attribution.to_parquet(
            os.path.join(args.experiment_dir, "code_attributions_interaction.parquet")
        )
        logger_main.log("Dump results")
