from argparse import ArgumentParser

import duckdb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

REGIONCODE_TO_NAME = {
    1084: "Capital Region Of Denmark",
    1085: "Region Zealand",
    1083: "Region of Southern Denmark",
    1082: "Central Denmark Region",
    1081: "North Denmark Region",
}
REGION_COLOR_MAP = {
    "Capital Region Of Denmark": "#585b70",
    "Region Zealand": "#c6a0f6",
    "Region of Southern Denmark": "#f8bd96",
    "Central Denmark Region": "#a6e3a1",
    "North Denmark Region": "#89dceb",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--duckdb_database", required=True, help="Path to duckdb database"
    )
    parser.add_argument(
        "--region_mapping",
        default="data/region_mapping.tsv",
        help="Path to mapping file between hospital and regions",
    )
    parser.add_argument(
        "--overwrite_region",
        action="store_true",
        help="Overwrite the current region mapping if it exists",
    )
    return parser.parse_args()


def column_exists(con, table, column):
    return (
        con.execute(f"DESCRIBE {table}")
        .pl()
        .get_column("column_name")
        .is_in(pl.lit(column))
        .any()
    )


def fetch_hospital_information(con) -> pl.DataFrame:
    """
    Takes a duckdb connection and returns hospital data for each patient in a polars dataframe
    The resulting dataframe has columns (pid: str, date: date, hospital_code: int).
    We filter events to A, B or C codes and must be in the period 1995-2018
    """

    sql = """
        SELECT PERSON_ID as pid, cast(C_SGH as int) as hospital_code,  D_INDDTO as date
              from t_diag_adm
              WHERE C_DIAGTYPE in ('A', 'B', 'C')
            AND YEAR(D_INDDTO) between 1995 AND 2018

    """
    return con.execute(sql).pl()


def valid_patients_with_region(con):
    query_valid_dates = f"""
        with joined_dataset as (
            select pid, modality, admit_date, region_code
            from dataset
            inner join (select pid, birthdate, future_outcome, outcome_date, region_code from patient_metadata) using (pid)
            
            where datediff('days', birthdate, admit_date)/365 > 0
            and datediff('days', admit_date, outcome_date)> 0
            and ((future_outcome and admit_date<outcome_date) or (not future_outcome and datediff('days', admit_date, outcome_date)>2*365))
            and extract(year from admit_date) between 1995 and 2018
            )
    
        select pid, cast(region_code as int) as region_code
        from (select pid, modality, min(admit_date) as min_date, max(admit_date) as max_date, region_code
              from joined_dataset 
              group by pid, modality, region_code
              having (modality = 'diag' and count(*) >= 1)
                or   (modality = 'prescription' and count(*) >= 5)
                or   (modality = 'ydelse' and count(*) >= 5)
              )
        group by pid, region_code
        having count(*) = 3
        order by pid
        """
    return con.execute(query_valid_dates).pl()


def main():
    args = parse_args()
    con = duckdb.connect(args.duckdb_database)
    hospital_to_region = (
        pl.read_csv(args.region_mapping, separator="\t")
        .select(pl.col("code").cast(pl.Int32), pl.col("region_code"))
        .unique()
    )
    hospital_events = fetch_hospital_information(con)
    hospital_events = hospital_events.join(
        hospital_to_region, left_on="hospital_code", right_on="code", how="left"
    )
    patient_region_visits = hospital_events.group_by(
        pl.col(["pid", "region_code"])
    ).count()
    filtered_patient_regions = patient_region_visits.join(
        patient_region_visits.group_by(pl.col("pid")).agg(
            max_count=pl.col("count").max()
        ),
        on="pid",
    ).filter(pl.col("max_count").eq(pl.col("count")))

    patients_single_assignments = (
        filtered_patient_regions.group_by(["pid", "max_count"])
        .count()
        .filter(pl.col("count").eq(1))
    )

    assigned_regions = (
        filtered_patient_regions.select("pid", "region_code")
        .filter(pl.col("pid").is_in(patients_single_assignments.get_column("pid")))
        .with_columns(region_assignment_type=pl.lit("single_highest"))
    )

    # Other patients are assigned based on region of last code
    tie_breaker_patients = (
        filtered_patient_regions.filter(
            pl.col("pid").is_in(patients_single_assignments.get_column("pid")).not_()
        )
        .get_column("pid")
        .unique()
    )

    tie_broken_regions = (
        hospital_events.filter(pl.col("pid").is_in(tie_breaker_patients))
        .group_by("pid")
        .agg(pl.col("region_code").sort_by("date").last())
    ).with_columns(region_assignment_type=pl.lit("tie-broken"))

    all_assigned_regions = pl.concat(
        [assigned_regions, tie_broken_regions]
    ).with_columns(pl.col("region_code").cast(pl.Int32))
    if (
        not column_exists(con, "patient_metadata", "region_code")
        or args.overwrite_region
    ):
        con.execute("""
        CREATE OR REPLACE TABLE patient_metadata as (
        SELECT * 
        from patient_metadata
        LEFT JOIN all_assigned_regions USING (pid)
        )
        """)
        print("Updated patient_metadata with region information")

    patients_with_region = valid_patients_with_region(con)
    count_regions = (
        patients_with_region.group_by(pl.col("region_code"))
        .agg(pl.count().alias("count"))
        .to_pandas()
    )

    count_regions["region_name"] = count_regions["region_code"].apply(
        lambda x: REGIONCODE_TO_NAME.get(x, "UNK")
    )

    fix, ax = plt.subplots()
    ax = count_regions.plot(kind="bar", x="region_name", y="count")
    ax.set_xlabel("Region")
    ax.set_ylabel("Number of patients")
    plt.tight_layout()

    plt.savefig("figures/validation/patients_per_region.pdf", dpi=300)
    plt.savefig("figures/validation/patients_per_region.svg", dpi=300)

    assigned_with_counts = (
        patients_with_region.select(
            pl.col("region_code").alias("assigned_region"), pl.col("pid")
        )
        .join(patient_region_visits, on="pid", how="inner")
        .to_pandas()
    )

    assigned_with_counts["assigned_region"] = assigned_with_counts[
        "assigned_region"
    ].apply(lambda x: REGIONCODE_TO_NAME.get(x, "NA"))
    assigned_with_counts["region"] = assigned_with_counts["region_code"].apply(
        lambda x: REGIONCODE_TO_NAME.get(x, "NA")
    )
    assigned_with_counts = assigned_with_counts[
        (assigned_with_counts["assigned_region"] != "NA")
        & (assigned_with_counts["region"] != "NA")
    ]

    # No clue how seaborn and facegrid works with hue and stacking so arguments are passed all over here
    g = sns.FacetGrid(
        assigned_with_counts,
        col="assigned_region",
        height=2.5,
        col_wrap=3,
    )
    g.map_dataframe(
        sns.histplot,
        x="count",
        log_scale=True,
        bins=30,
        multiple="stack",
        hue="region",
        palette=REGION_COLOR_MAP,
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels(x_var="Number of hospital codes", y_var="Number of patients")
    legend = {k: mpatches.Patch(color=v, label=k) for k, v in REGION_COLOR_MAP.items()}

    g.add_legend(legend_data=legend)
    g.tight_layout()

    plt.savefig(
        "figures/validation/region_assignment_relative_distributions.pdf", dpi=300
    )
    plt.savefig(
        "figures/validation/region_assignment_relative_distributions.svg", dpi=300
    )

    relative_counts_per_region = (
        assigned_with_counts.groupby(["assigned_region", "region"])["count"]
        .sum()
        .reset_index(drop=False)
    )
    total_region_count = (
        relative_counts_per_region.groupby(("assigned_region"))["count"]
        .sum()
        .reset_index(drop=False)
    )

    relative_counts_per_region = relative_counts_per_region.merge(
        total_region_count, on="assigned_region"
    )
    relative_counts_per_region["normalized"] = (
        relative_counts_per_region["count_x"] / relative_counts_per_region["count_y"]
    ) * 100

    fix, ax = plt.subplots()
    ax = sns.histplot(
        relative_counts_per_region,
        weights="normalized",
        x="assigned_region",
        multiple="stack",
        hue="region",
        palette=REGION_COLOR_MAP,
    )
    plt.xticks(rotation=90)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("Percentage of codes")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig("figures/validation/region_assignment_relative_counts.pdf", dpi=300)
    plt.savefig("figures/validation/region_assignment_relative_counts.svg", dpi=300)

    con.close()


if __name__ == "__main__":
    main()
