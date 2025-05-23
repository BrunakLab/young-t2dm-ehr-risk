import argparse
import json
import math
import os
import sys
from functools import partial
from multiprocessing import Pool
from os.path import dirname, realpath

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import sklearn.metrics
from sklearn.metrics._ranking import _binary_clf_curve

sys.path.insert(0, dirname(dirname(realpath(__file__))))
from diabnet.utils.eval import get_probs_golds


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    k = 1 - confidence
    k *= 100
    hm, m, mh = np.percentile(a, (k / 2, 50, 100 - k / 2))
    return {"ci_low": hm, "Median": m, "ci_high": mh}


def calc_relative_risk(tp, tn, fp, fn):
    precision = tp / (tp + fp)
    incidence = (tp + fn) / (tp + fp + fn + tn)
    return precision / incidence


def get_boot_metric_clf(seed, n, probs, golds):
    np.random.seed(seed)
    results = []
    for _ in range(n):
        sample = np.random.randint(len(probs), size=len(probs))
        resampled_probs = probs[sample]
        resampled_golds = golds[sample]
        fps, tps, thresholds = _binary_clf_curve(
            resampled_golds, resampled_probs, pos_label=1
        )

        if len(thresholds) == 1:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        p_count = tps[-1]
        n_count = fps[-1]

        fns = p_count - tps
        tns = n_count - fps
        precisions = tps / (tps + fps)
        precisions[np.isnan(precisions)] = 0
        recalls = tps / p_count

        fprs = fps / n_count
        tprs = tps / p_count

        np.seterr(invalid="ignore")
        with np.errstate(divide="ignore"):
            odds_ratio = np.nan_to_num(
                (tps / fps) / np.nan_to_num(fns / tns), posinf=0, nan=0
            )
        ps = tps + fps
        f1s = 2 * tps / (ps + p_count)
        incidence_ = p_count / (p_count + n_count)

        auprc_ = sklearn.metrics.auc(recalls, precisions)
        auroc_ = sklearn.metrics.auc(fprs, tprs)

        if args.index == "f1":
            risk_index = np.nanargmax(f1s)
        elif args.index == "0.1":
            risk_index = int(1000 / 1e6 * len(resampled_probs))
        else:
            try:
                risk_index = (precisions >= 0.05).nonzero()[0].max()
            except ValueError:
                print("No index with over 0.05 precision. Setting index to 1")
                risk_index = 1

        precision_ = precisions[risk_index]
        recall_ = recalls[risk_index]
        odds_ratio_ = odds_ratio[risk_index]
        tpr_ = tprs[risk_index]
        fpr_ = fprs[risk_index]
        threshold_ = thresholds[risk_index]

        pauc_ = sklearn.metrics.roc_auc_score(
            y_true=resampled_golds, y_score=resampled_probs, max_fpr=0.1
        )
        tn, fp, fn, tp = (
            tns[risk_index],
            fps[risk_index],
            fns[risk_index],
            tps[risk_index],
        )
        relative_risk_ = calc_relative_risk(tp, tn, fp, fn)
        mcc_ = sklearn.metrics.matthews_corrcoef(
            golds_for_eval, (probs_for_eval > threshold_).astype(np.int8)
        )
        results.append(
            {
                "auroc": auroc_,
                "fpr": fpr_,
                "tpr": tpr_,
                "auprc": auprc_,
                "precision": precision_,
                "recall": recall_,
                "mcc": mcc_,
                "relative_risk": relative_risk_,
                "odds_ratio": odds_ratio_,
                "incidence": incidence_,
                "threshold": threshold_,
                "pauc": pauc_,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return results


def get_performance_ci(
    probs_for_eval,
    golds_for_eval,
    cores,
    n_boot,
):
    with Pool(cores) as pool:
        bootstraps = pool.map(
            partial(
                get_boot_metric_clf,
                probs=probs_for_eval,
                golds=golds_for_eval,
                n=math.ceil(n_boot // cores),
            ),
            range(cores),
        )
    bootstraps = [result for worker in bootstraps for result in worker]

    metrics = {}
    for bootstrap in bootstraps:
        for metric, value in bootstrap.items():
            if metric in metrics:
                metrics[metric].append(value)
            else:
                metrics[metric] = [value]

    confidence_intervals = []
    for metric, values in metrics.items():
        record = {"Metric": metric}
        record.update(mean_confidence_interval(values))
        confidence_intervals.append(record)

    return confidence_intervals


def get_slice(
    df,
    model_name=None,
    metric_name=None,
    modality=None,
    prediction_interval=None,
    exclusion_interval=None,
):
    if model_name is not None:
        df = df.loc[df.Model == model_name]
    if metric_name is not None:
        if type(metric_name) is str:
            df = df.loc[df.Metric == metric_name]
        else:
            df = df.loc[[i in metric_name for i in df.Metric]]
    if modality is not None:
        df = df.loc[df["modality"] == modality]
    if prediction_interval is not None:
        df = df.loc[df["Prediction Interval"] == prediction_interval]
    if exclusion_interval is not None:
        df = df.loc[df["Exclusion Interval"] == exclusion_interval]
    return df


def filter_positive_patients_negative_trajectories(patient_outcome, trajectory_outcome):
    return patient_outcome == trajectory_outcome


def filter_complete_negative_trajectories(patient_outcome):
    return patient_outcome == 1


def filter_positive_window(patient_outcome, time_to_outcome, minimum_time_to_outcome):
    return (patient_outcome != 1) | (time_to_outcome > minimum_time_to_outcome)


def get_complication_patients(
    return_true: bool,
    years_to_complication: int,
    database,
):
    conn = duckdb.connect(database, read_only=True)
    metadata = conn.execute(
        f"SELECT * from patient_metadata WHERE future_outcome == 1 AND datediff('year', birthdate, outcome_date) < 40",
    ).pl()
    complications = conn.execute(
        """SELECT PERSON_ID as pid, C_DIAG as code, D_INDDTO as admit_date 
             from 
                t_diag_adm 
             where 
                C_DIAG like 'DE10%'
             OR C_DIAG like 'DE11%'
             OR C_DIAG like 'DE12%'
             OR C_DIAG like 'DE13%'
             OR C_DIAG like 'DE14%'
                    """,
    ).pl()
    conn.close()
    outcome_data = (
        metadata.select(
            pl.col(["pid", "sex", "birthdate", "outcome_date", "future_outcome"])
        )
        .join(complications, on="pid")
        .filter(
            pl.col("admit_date")
            .sub(pl.duration(days=years_to_complication * 365))
            .le(pl.col("outcome_date"))
        )
        .with_columns(code=pl.col("code").str.slice(0, 5))
        .group_by(["pid", "code"])
        .count()
        .with_columns(
            is_complication=pl.col("code").str.contains(r"^DE1[0-4][0-8]$"),
        )
        .group_by("pid")
        .agg(
            complication_status=pl.col("is_complication").any(),
        )
    )
    if not return_true:
        return outcome_data.filter(pl.col("complication_status").not_()).to_pandas()
    return outcome_data.filter(pl.col("complication_status")).to_pandas()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PancNet Grid Search Results Collector."
    )
    parser.add_argument(
        "--performance_table",
        required=True,
        type=str,
        help="Path to the performance table generated from step 3",
    )
    parser.add_argument(
        "--bootstrap_size", type=int, default=200, help="Number of bootstraps"
    )
    parser.add_argument(
        "--cores", type=int, default=16, help="Number of cores available"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Downsample the output prediction. 1 for every N samples.",
    )
    parser.add_argument(
        "--filter_time_negatives",
        action="store_true",
        default=False,
        help="Exclude trajectories from positive patients that are prior to endpoint and therefore negative",
    )
    parser.add_argument(
        "--filter_complete_negatives",
        action="store_true",
        default=False,
        help="Exclude trajectories from patient without outcome",
    )
    parser.add_argument(
        "--filter_positive_window_difference",
        action="store_true",
        help="Size of positive window size. Removes trajectories close to outcome (i.e. for endpoint 12 with size 3 removes trajectories 9 months prior to outcome)",
    )
    parser.add_argument(
        "--inclusion_by_file",
        help="csv file with a pid column to indicate the pids that should be used as a subpopulation for calculating estimates.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="bootstrap",
        help="Prefix for filenames",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="figures",
        help="Name of directory in --search_metadata to save outputs to",
    )
    parser.add_argument(
        "--index",
        choices=["precision", "0.1", "f1"],
        default="0.1",
        help="How should the index for the positive threshold be deterimned",
    )
    parser.add_argument(
        "--sex",
        default=None,
        type=int,
        choices=[1, 2, None],
        help="Run metrics for only a single sex (1: Male, 2: Female, None: both) (default: None)",
    )
    parser.add_argument(
        "--complication",
        default=None,
        type=int,
        choices=[0, 1, None],
        help="Run metrics for only patients with or without complications (0: No complications, 1: Complications) (default: None)",
    )
    parser.add_argument(
        "--metadata-db",
        help="What duckdb should the data about sex be fetched from",
    )

    args = parser.parse_args()

    assert args.bootstrap_size > 1, "Choose a boot size higher than 1. "
    best_exp_ids_config = pd.read_csv(args.performance_table)
    best_exp_ids_config = best_exp_ids_config.drop_duplicates()
    best_exp_ids_config = best_exp_ids_config.to_dict("list")
    prefix = "{}_TableS4".format(args.filename)

    metrics_records = []
    for i, (experiment_dir, model_name, modality, exclusion_interval) in enumerate(
        zip(
            best_exp_ids_config["experiment_dir"],
            best_exp_ids_config["encoder"],
            best_exp_ids_config["modalities"],
            best_exp_ids_config["exclusion_interval"],
        )
    ):
        printing_prefix = "[Step4-ResultsBootstrap][{}/{}]".format(
            i + 1, len(best_exp_ids_config["experiment_dir"])
        )
        test_preds_path = os.path.join(experiment_dir, "test_preds.parquet")
        if not os.path.exists(test_preds_path):
            print(
                printing_prefix,
                "[WARNING] File not found at {} ! Consider not to skip step 4.".format(
                    test_preds_path
                ),
            )
            continue
        test_preds = pd.read_parquet(test_preds_path)
        params = json.load(open(os.path.join(experiment_dir, "args.json"), "r"))
        print(printing_prefix, "[INFO] Data loaded from {}... ".format(test_preds_path))
        for index, prediction_interval in enumerate(params["month_endpoints"]):
            print(
                printing_prefix,
                "Processing time interval: {} [{}/{}].".format(
                    prediction_interval, index + 1, len(params["month_endpoints"])
                ),
            )
            if exclusion_interval >= prediction_interval:
                print(
                    printing_prefix,
                    f"No positives for prediction interval {prediction_interval} with exclusion interval {exclusion_interval}",
                )
                continue
            probs_for_eval, golds_for_eval, trajectory_mask = get_probs_golds(
                test_preds, index=index, return_mask=True
            )
            probs_for_eval = np.array(probs_for_eval)[:: args.n_samples]
            golds_for_eval = np.array(golds_for_eval)[:: args.n_samples]
            trajectory_mask = np.array(trajectory_mask)[:: args.n_samples]
            trajectory_mask_filter = np.full(probs_for_eval.shape, fill_value=True)

            if args.filter_time_negatives:
                patient_outcome = np.array(test_preds["patient_golds"])[
                    :: args.n_samples
                ]
                patient_outcome = patient_outcome[trajectory_mask]
                outcome_mask = filter_positive_patients_negative_trajectories(
                    patient_outcome, golds_for_eval
                )
                print(
                    printing_prefix,
                    f"Filtering {(outcome_mask == False).sum()} that are negative trajectories from positive patients",
                )
                trajectory_mask_filter = trajectory_mask_filter & outcome_mask

            if args.filter_complete_negatives:
                patient_outcome = np.array(test_preds["patient_golds"])[
                    :: args.n_samples
                ]
                patient_outcome = patient_outcome[trajectory_mask]
                outcome_mask = filter_complete_negative_trajectories(patient_outcome)
                print(
                    printing_prefix,
                    f"Filtering {(outcome_mask == False).sum()} that are negative trajectories from completely negative patients",
                )
                trajectory_mask_filter = trajectory_mask_filter & outcome_mask

            if args.filter_positive_window_difference:
                patient_outcome = np.array(test_preds["patient_golds"])[
                    :: args.n_samples
                ]
                patient_outcome = patient_outcome[trajectory_mask]

                time_to_outcome = np.array(test_preds["days_to_final_censors"])[
                    :: args.n_samples
                ]
                time_to_outcome = time_to_outcome[trajectory_mask]
                positive_difference = (
                    prediction_interval - params["month_endpoints"][0]
                ) * 30

                window_mask = filter_positive_window(
                    patient_outcome, time_to_outcome, positive_difference
                )
                print(
                    printing_prefix,
                    f"Filtering {(window_mask == False).sum()} positive trajectories that are closer than {positive_difference / 30} months to outcome",
                )
                trajectory_mask_filter = trajectory_mask_filter & window_mask

            if args.sex:
                con = duckdb.connect(args.metadata_db, read_only=True)
                patients_keep = con.execute(
                    """
                SELECT pid from patient_metadata
                WHERE sex = ?
                """,
                    [args.sex],
                ).fetchnumpy()["pid"]
                con.close()
                patients = test_preds["pids"]
                patients = patients[trajectory_mask]
                sex_mask = np.isin(patients, patients_keep)
                trajectory_mask_filter = trajectory_mask_filter & sex_mask
                print(
                    printing_prefix,
                    f"Filtering {(sex_mask == False).sum()} based on sex",
                )

            if args.complication != None:
                complication_patients = get_complication_patients(
                    args.complication == 1, 5, args.metadata_db
                )
                patients = test_preds["pids"][trajectory_mask]
                patient_golds = test_preds["patient_golds"][trajectory_mask]
                dates = test_preds["dates"][trajectory_mask]
                complication_mask = (
                    np.isin(patients, complication_patients["pid"])
                    | (patient_golds[trajectory_mask] == 0)
                ) & (dates < "2013-12-31")
                trajectory_mask_filter = trajectory_mask_filter & complication_mask
                print(
                    printing_prefix,
                    f"Filtering {(complication_mask == False).sum()} based on complication",
                )
            if args.inclusion_by_file:
                pids_to_keep = pd.read_csv(args.inclusion_by_file)["pid"].values
                patients = test_preds["pids"][trajectory_mask]
                inclusion_mask = np.isin(patients, pids_to_keep)
                trajectory_mask_filter = trajectory_mask_filter & inclusion_mask
                print(
                    printing_prefix,
                    f"Filtering {(inclusion_mask == False).sum()} based on inclusion file",
                )

            probs_for_eval = probs_for_eval[trajectory_mask_filter]
            golds_for_eval = golds_for_eval[trajectory_mask_filter]
            if not np.sum(golds_for_eval) > 0:
                continue
            metrics = get_performance_ci(
                probs_for_eval,
                golds_for_eval,
                n_boot=args.bootstrap_size,
                cores=args.cores,
            )
            experiment_info = {
                "Model": model_name,
                "Modality": modality,
                "Prediction Interval": prediction_interval,
                "Exclusion Interval": exclusion_interval,
                "experiment_dir": experiment_dir,
            }
            [metric.update(experiment_info) for metric in metrics]

            metrics_records.extend(metrics)

    os.makedirs(
        os.path.join(os.path.dirname(args.performance_table), args.output_dir_name),
        exist_ok=True,
    )
    os.chdir(
        os.path.join(os.path.dirname(args.performance_table), args.output_dir_name)
    )

    df = pd.DataFrame.from_records(metrics_records)
    df = df.astype({"Prediction Interval": "int32", "Exclusion Interval": "int32"})
    df.to_csv(prefix + ".Performance_table.csv", sep=",", index=False)

    df["print_aucs"] = [
        (
            "{:.3f} ({:.3f}-{:.3f})".format(i["Median"], i["ci_low"], i["ci_high"])
            if i["Metric"][:2] == "au"
            else np.nan
        )
        for i in df.iloc
    ]
    df["print_specificity"] = [
        (
            "{:.2%} ({:.2%}-{:.2%})".format(
                1 - i["Median"], 1 - i["ci_high"], 1 - i["ci_low"]
            )
            if i["Metric"] == "fpr"
            else np.nan
        )
        for i in df.iloc
    ]
    df["print_others"] = [
        (
            "{:.1%} ({:.1%}-{:.1%})".format(i["Median"], i["ci_low"], i["ci_high"])
            if i["Metric"][:2] not in ["au", "fp"] and i["Metric"] not in ["curves"]
            else np.nan
        )
        for i in df.iloc
    ]
    df["print_merged"] = [
        i["print_specificity"] if i["Metric"] == "fpr" else i["print_others"]
        for i in df.iloc
    ]

    get_slice(df, metric_name="auroc").pivot_table(
        values="print_aucs",
        columns=["Prediction Interval"],
        index=["Model", "Modality", "Exclusion Interval"],
        aggfunc=lambda x: [v for v in x],
    ).fillna("-").to_csv(prefix + ".Performance_summary_auroc.csv")

    get_slice(df, metric_name="auprc").pivot_table(
        values="print_aucs",
        columns=["Prediction Interval"],
        index=["Model", "Modality", "Exclusion Interval"],
        aggfunc=lambda x: [v for v in x],
    ).fillna("-").to_csv(prefix + ".Performance_summary_auprc.csv")

    get_slice(df, metric_name=["precision", "recall"]).pivot_table(
        values="print_others",
        columns=["Prediction Interval", "Metric"],
        index=["Model", "Modality", "Exclusion Interval"],
        aggfunc=lambda x: [v for v in x],
    ).fillna("-").to_csv(prefix + ".Performance_summary_pr.csv")

    get_slice(df, metric_name=["fpr", "precision", "recall"]).pivot_table(
        values="print_merged",
        index=["Model", "Modality", "Exclusion Interval", "Metric"],
        columns=["Prediction Interval"],
        aggfunc=lambda x: [v for v in x][0],
    ).fillna("-").to_csv(prefix + ".Performance_summary_p-r-s.csv")
