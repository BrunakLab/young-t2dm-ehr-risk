# This scripts is used to collect and parse the available results of a single search, the performance is summarized in a
# summary file.
import argparse
import json
import os
import pickle as pkl
import re
import sys
from os.path import dirname, realpath

import pandas as pd

sys.path.insert(0, dirname(dirname(realpath(__file__))))
import diabnet.utils.parsing as parsing
from diabnet.learn.state_keeper import STATS_PATH

SUMMARIZING_MSG = (
    "[Step3-CollectSearchResults][1/3][INFO] Summarizing results {} files into {}."
)

GROUPING_ARGS = ["encoder", "exclusion_interval", "modalities"]
PARAMS_TO_SAVE = ["experiment_dir"] + GROUPING_ARGS

parser = argparse.ArgumentParser(description="PancNet Grid Search Results Collector.")
parser.add_argument(
    "--experiment_dir",
    type=str,
    required=True,
    help="Where logs and result files from step 2 is stored.",
)
parser.add_argument(
    "--summary_dir",
    default=None,
    help="Where to store the information for step 3. Will concatenate results to already existing gathered results. Default is the experiment dir",
)
parser.add_argument(
    "--metric",
    type=str,
    default="auprc",
    help="Metric to use for ordering the results.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="Overwrite existing summary file.",
)

parser.add_argument(
    "--timepoint",
    action="store",
    default=36,
    type=int,
    help="Timepoint to evaluate metrics from.",
)
args = parser.parse_args()
args.summary_dir = args.summary_dir if args.summary_dir else args.experiment_dir


def get_metric_key(timepoint, metric):
    return f"dev_{timepoint}month_{metric}"


def write_job_list(job_list, filename):
    with open(filename, "w") as out_file:
        out_file.write("{}\n".format(config_file))
        for job in job_list:
            if "--" in job:
                job_md5 = parsing.md5(job)
            else:
                job_md5 = job
            out_file.write("{}, {}\n".format(job_md5, job))


def readjust_metric_labels(stats, exclusion_interval):
    old_labels, new_labels = [], []
    for label in stats.keys():
        result = re.search(r"\d+", label)
        if result:
            prediction_month = int(result.group())
            new_label = re.sub(
                r"\d+", str(prediction_month - exclusion_interval), label
            )
            old_labels.append(label)
            new_labels.append(new_label)
    return old_labels, new_labels


def update_summary_with_results(result_dir):
    job_ids = os.listdir(result_dir)
    summary = []
    for job in job_ids:
        job_path = os.path.join(result_dir, job)
        stats_path = os.path.join(job_path, STATS_PATH)
        print(stats_path)
        args_path = os.path.join(job_path, "args.json")
        print(
            "[Step3-CollectSearchResults][2/3][INFO] Updating summary with result {}".format(
                job_path
            )
        )

        try:
            results = pkl.load(open(stats_path, "rb"))
            params = json.load(open(args_path, "r"))

            if params["relative_endpoints"]:
                old_labels, new_labels = readjust_metric_labels(
                    results, params["exclusion_interval"]
                )
                for old_label, new_label in zip(old_labels, new_labels):
                    results[new_label] = results.pop(old_label)

        except FileNotFoundError as er:
            print(er)
            print(
                "[Step3-CollectSearchResults][2/3][ERR] Experiment failed or the result file is in another location! "
                "Failed job: {}".format(job)
            )
            continue

        if (
            params["exclusion_interval"] >= args.timepoint
            and not params["relative_endpoints"]
        ):
            print(
                f"[Step3-CollectSearchResults][2/3][WARN] Skipping {job} : Exclusion interval is smaller than month endpoint"
            )
            continue

        best_epoch_results = {
            k: v[results["best_epoch"]] for k, v in results.items() if k != "best_epoch"
        }
        best_epoch_results.update(
            {k: v for k, v in params.items() if k in PARAMS_TO_SAVE}
        )
        summary.append(best_epoch_results)
    return summary


def extract_best_models(summary):
    score_metric = get_metric_key(args.timepoint, args.metric)
    summary["metric"] = score_metric
    try:
        idx = (
            summary.groupby(GROUPING_ARGS)[score_metric].transform("max")
            == summary[score_metric]
        )
    except KeyError as err:
        print(err)
        print(
            f"Please ensure that metric {score_metric} exists. Can be modified through --metric and --timepoint"
        )
        sys.exit(1)

    filtered_models = summary[idx]
    return filtered_models


if __name__ == "__main__":
    print("[Step3-CollectSearchResults][1/3]Start to collect grid exprs...")
    config_file = os.path.join(args.experiment_dir, "search_config.json")

    assert os.path.exists(args.experiment_dir)
    os.makedirs(args.summary_dir, exist_ok=True)

    experiment_config_json = json.load(open(config_file, "r"))
    job_list = parsing.parse_dispatcher_config(experiment_config_json)
    job_ids = [parsing.md5(job) for job in job_list]
    master_id = parsing.md5("".join(job_list))

    jobfile = args.summary_dir + "/joblist_{}.txt".format(master_id)
    if os.path.exists(jobfile) and not args.overwrite:
        print("Aborting to avoid overwriting...")
        sys.exit(1)

    print(
        SUMMARIZING_MSG.format(
            len(job_list), args.summary_dir + "/master.{}.summary".format(master_id)
        )
    )

    write_job_list(job_list, jobfile)

    print("[Step3-CollectSearchResults][2/3] Start to load result files...")
    result_dir = os.path.join(args.experiment_dir, "results")
    summary = update_summary_with_results(result_dir)
    summary = pd.DataFrame.from_dict(summary)
    summary["modalities"] = summary["modalities"].apply(lambda x: "_".join(x))
    summary["master_id"] = master_id
    print("[Step3-CollectSearchResults][3/3] Start exporting... ")

    raw_summary_path = os.path.join(args.summary_dir, "raw_models.csv")
    summary.to_csv(
        raw_summary_path, mode="a", header=not os.path.exists(raw_summary_path)
    )

    best_model_summary = extract_best_models(summary)
    best_summary_path = os.path.join(args.summary_dir, "best_models.csv")
    best_model_summary.to_csv(
        best_summary_path, mode="a", header=not os.path.exists(best_summary_path)
    )
