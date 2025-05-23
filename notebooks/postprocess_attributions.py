import sys
from argparse import ArgumentParser
from pathlib import Path

import duckdb
import polars as pl

sys.path.append(str(Path(__file__).parents[2]))

CODE_N_CUTOFF = 1000  # N patients that need to have


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "run_dir",
        help="Directory with test_preds.parquet, attributions.parquet and interaction_attributions.parquet",
    )
    parser.add_argument(
            "--duckdb_database",
            help="Duckdb database containing the dataset and metadata",
        )
    return parser.parse_args()


def load_documentation():
    atc = pl.read_csv("data/documentation/atc_descriptions.csv")
    icd = pl.read_csv("data/documentation/icd10_disease_descriptions.tsv", separator="\t")
    ydelse = pl.read_csv(
        "data/documentation/ydelse_descriptions.tsv",
        separator="\t",
    )

    atc = atc.select(
        pl.col("atc_code").alias("code"),
        pl.col("atc_name").alias("description"),
        pl.col("atc_code").str.slice(0, 1).alias("chapter_code"),
        pl.lit("prescription").alias("modality"),
    )
    atc = atc.join(
        atc.select(pl.col("code").alias("join_code"), pl.col("description").alias("chapter")),
        left_on="chapter_code",
        right_on="join_code",
    )
    atc = atc.select(pl.col(["code", "description", "chapter", "modality"]))

    return pl.concat(
        [
            atc,
            icd.select(
                pl.concat_str(pl.lit("D"), pl.col("code")).alias("code"),
                pl.col("level3_description").alias("description"),
                pl.col("level1_num").alias("chapter"),
                pl.lit("diag").alias("modality"),
            ),
            ydelse.select(
                pl.col("fullcode_grouped").alias("code").cast(pl.Utf8),
                pl.col("ydelse_description").alias("description"),
                pl.col("merg_descr").str.split("~").list.last().alias("chapter"),
                pl.lit("ydelse").alias("modality"),
            ),
        ]
    ).unique()


documentation = load_documentation()


def extract_patient_attributions(
    attributions,
    maximum_days_prior_assessment: int,
    codes,
    groupby_cols=None,
    time_to_censor_lower: int = 0,
    time_to_censor_upper: int = 10000,
):
    if groupby_cols is None:
        groupby_cols = []
    elif isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]
    elif not isinstance(groupby_cols, list):
        raise TypeError()

    filtered_attributions = attributions.filter(
        pl.col("time_to_trajectory_end").lt(maximum_days_prior_assessment),
        pl.col("days_to_censor").is_between(time_to_censor_lower, time_to_censor_upper),
        pl.col("code").is_in(codes),
    )

    patient_level = filtered_attributions.group_by(
        pl.col(
            groupby_cols + ["pid", "birthdate", "outcome_date", "days_to_censor", "code", "sex"]
        )
    ).agg(pl.col("attribution").sum(), pl.count().alias("count"))
    code_level = (
        patient_level.group_by(pl.col(groupby_cols + ["code"]))
        .agg(
            patient_average_attribution=pl.col("attribution").mean(),
            total_sum_attribution=pl.col("attribution").sum(),
            code_counts=pl.col("count").sum(),
            patient_counts=pl.count(),
        )
        .collect()
    )
    return patient_level.collect().join(documentation, on="code"), code_level.join(
        documentation, on="code"
    )


if __name__ == "__main__":

    def main():
        args = parse_args()
        DIR = Path(args.run_dir)

        con = duckdb.connect(args.duckdb_database , read_only=True)
        code_frequencies = con.execute(""" SELECT truncated_code as code, count(*) as count
                        FROM (SELECT
                                CASE 
                                    WHEN modality = 'ydelse' THEN code
                                    WHEN modality = 'prescription' THEN substring (code, 1, 5)
                                    WHEN modality = 'diag' THEN substring (code, 1, 4)
                                END as truncated_code
                            from combined_dataset
                            )
                        GROUP BY truncated_code""").pl()
        codes = code_frequencies.filter(pl.col("count").ge(CODE_N_CUTOFF)).get_column("code")

        test_preds = pl.scan_parquet(DIR / "test_preds.parquet").with_columns(
            pl.col("probs").list.get(-1),
            pl.col("days_to_final_censors").list.get(0),
            pl.col("dates").str.to_date(),
        )
        pids_of_interest = test_preds.select(pl.col("pids").unique()).collect().to_series()

        metadata = (
            con.execute(
                f"SELECT * from patient_metadata where pid in ({' ,'.join(['?' for _ in range(len(pids_of_interest))])})",
                pids_of_interest.to_list(),
            )
            .pl()
            .lazy()
        )
        attributions = pl.scan_parquet(DIR / "code_attributions.parquet").join(metadata, on="pid")
        attributions_interaction = pl.scan_parquet(
            DIR / "code_attributions_interaction.parquet"
        ).join(metadata, on="pid")

        all_positive_distribution, all_positive_attribution = extract_patient_attributions(
            attributions.filter(pl.col("future_outcome").eq(1)), 365 * 10, codes
        )
        positive_distribution_interaction, positive_attribution_interaction = (
            extract_patient_attributions(
                attributions_interaction.filter(pl.col("future_outcome").eq(1)),
                365 * 10,
                codes,
                groupby_cols=["removed_code"],
            )
        )
        positive_distribution_interaction = positive_distribution_interaction.with_columns(
            removed_code=pl.when(pl.col("removed_code").str.contains("^D[A-Z]"))
            .then(pl.col("removed_code").str.slice(0, 4))
            .when(pl.col("removed_code").str.contains(r"^[^\d]"))
            .then(pl.col("removed_code").str.slice(0, 5))
            .otherwise(pl.col("removed_code"))
        ).join(documentation, left_on="removed_code", right_on="code", suffix="_removed")

        negative_distributions, negative_attributions = extract_patient_attributions(
            attributions.filter(pl.col("future_outcome").eq(0)), 365 * 10, codes
        )

        code_attributions_by_year = []
        distributions = []
        n_years = 5
        for year in range(n_years):
            print(f"Calculating {year + 1}/{n_years}")
            positive_distributions, positive_attributions = extract_patient_attributions(
                attributions.filter(pl.col("future_outcome").eq(1)),
                365 * 10,
                codes,
                time_to_censor_lower=365 * year,
                time_to_censor_upper=365 * (year + 1),
            )
            distributions.extend(
                [positive_distributions.with_columns(years_to_diagnosis=pl.lit(year))]
            )
            code_attributions_by_year.extend(
                [positive_attributions.with_columns(years_to_diagnosis=pl.lit(year))]
            )

        distributions = pl.concat(distributions)
        code_attributions_by_year = pl.concat(code_attributions_by_year)

        negative_distributions.write_parquet(DIR / "negative_attributions_patients.parquet")
        code_attributions_by_year.write_parquet(DIR / "positive_attributions_yearly.parquet")
        distributions.write_parquet(DIR / "positive_attributions_patients_yearly.parquet")

        all_positive_attribution.write_parquet(DIR / "positive_attributions.parquet")
        all_positive_distribution.write_parquet(DIR / "positive_attributions_patients.parquet")
        positive_distribution_interaction.write_parquet(
            DIR / "positive_attributions_interactions.parquet"
        )

    if __name__ == "__main__":
        main()
