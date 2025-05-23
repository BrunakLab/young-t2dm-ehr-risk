from argparse import ArgumentParser

import matplotlib.pyplot as plt
import polars as pl


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prediction_path", help="Path to test_preds.parquet file")
    parser.add_argument(
        "--timepoint",
        type=int,
        help="Index of which time point to use (default is first(0))",
        default=0,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preds = pl.scan_parquet(args.prediction_path)
    threshold = (
        preds.with_columns(pl.col("probs").list.get(args.timepoint))
        .sort("probs", descending=True)
        .with_columns(
            pl.col("golds").cumsum().alias("tp_cumsum"),
            pl.arange(1, pl.count() + 1).alias("total_cumsum"),
        )
        .with_columns(precision=pl.col("tp_cumsum") / pl.col("total_cumsum"))
        .filter(pl.col("precision").gt(0.05))
        .select("probs")
        .min()
        .collect()
        .item()
    )
    positives = preds.filter(
        pl.col("probs").list.get(args.timepoint).ge(pl.lit(threshold))
        & pl.col("patient_golds").eq(1)
    ).collect()

    positive_times = (
        positives.select(
            pl.col("pids"), pl.col("days_to_final_censors").list.get(0).cast(pl.Int32)
        )
        .filter(pl.col("days_to_final_censors").le(2 * 365))
        .group_by("pids")
        .agg(pl.col("days_to_final_censors").max())
    )

    negative_times = (
        preds.filter(
            pl.col("patient_golds").eq(1)
            & pl.col("pids").is_in(positives.get_column("pids")).not_()
        )
        .select("pids")
        .unique()
        .collect()
    ).with_columns(days_to_final_censors=pl.lit(0))

    average_days_benefit = (
        pl.concat([positive_times, negative_times])
        .select(pl.col("days_to_final_censors").mean())
        .item()
    )
    print(
        f"The estimated potential benefit in days diagnosed earlier:",
        average_days_benefit,
    )
    days_benefit = pl.concat([positive_times, negative_times]).to_pandas()

    days_benefit["days_to_final_censors"].plot()
    plt.figure(figsize=(10, 6))
    plt.hist(
        days_benefit["days_to_final_censors"],
        bins=50,
        color="skyblue",
        edgecolor="black",
    )

    plt.xlabel("Potential Reduction in Diagnosis Time")
    plt.ylabel("Number of Patients")
    plt.grid(True)
    plt.savefig("figures/diagnostic_delay/potential_diagnostic_benefit.pdf")


if __name__ == "__main__":
    main()
