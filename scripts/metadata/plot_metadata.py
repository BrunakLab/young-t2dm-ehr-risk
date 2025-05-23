import json
import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path

sys.path.append("./")
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set_theme(style="whitegrid")

logger = Logger("Metadata Logger")


def parse_job_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--run_dir",
        action="store",
        type=str,
        default=None,
        help="Directory of the experiment dir (requires a args.json to be defined here)",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        required=True,
        help="Path to output directory for figures and pickled metadata",
    )
    parser.add_argument(
        "--overwrite_metadata",
        action="store_true",
        default=False,
        help="Should any previous created metadata.pkl file in the output directory be deleted",
    )
    return parser.parse_args()


def get_complications_positive_patients(
    conn,
    pids,
    years_to_complication: int,
):
    metadata = conn.execute(
        f"""SELECT * from patient_metadata 
        where pid in (select pid from pids)
        AND future_outcome == 1
        AND datediff('year', birthdate, outcome_date) < 40""",
    ).pl()
    complications = conn.execute(
        """SELECT PERSON_ID as pid, C_DIAG as code, D_INDDTO as admit_date
             from 
                t_diag_adm 
             where PERSON_ID in (select pid from pids)
            AND (C_DIAG like 'DE10%'
             OR C_DIAG like 'DE11%'
             OR C_DIAG like 'DE12%'
             OR C_DIAG like 'DE13%'
             OR C_DIAG like 'DE14%')
                    """,
    ).pl()

    complication_data = (
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
    return complication_data.to_pandas()


def format_metadata_sql(
    min_age_assessment,
    days_exclusion,
    min_year_admission,
    max_year_admission,
    min_events_diag,
    min_events_prescription,
    min_events_ydelse,
    min_events_length,
):
    filtering_sql = f"""
    datediff('days', birthdate, admit_date)/365 > {min_age_assessment}
    and datediff('days', admit_date, outcome_date)>={days_exclusion}
    and ((future_outcome and admit_date<=outcome_date) 
            or (not future_outcome and datediff('days', admit_date, outcome_date)>=2*365))
    and extract(year from admit_date) between {min_year_admission} and {max_year_admission}"""

    all_pids_sql = f"""
    select pid 
    from (
        select pid, modality
        from dataset
        inner join (select pid, birthdate, future_outcome, outcome_date from patient_metadata) using (pid)
        where {filtering_sql}
        group by pid, modality
        having (modality = 'diag' and count(*) >= {min_events_diag})
            or   (modality = 'prescription' and count(*) >= {min_events_prescription})
            or   (modality = 'ydelse' and count(*) >= {min_events_ydelse})
        )
    group by pid
    having count(*) = 3
    """

    all_valid_events_sql = f"""
    select *
    from dataset
    inner join (select pid, sex, birthdate, future_outcome, split_group, region_code, outcome_date from patient_metadata) using (pid)
    where pid in (select pid from pids)
    and {filtering_sql}
    """

    codes_by_year_sql = f"""
    select code, modality, future_outcome as outcome, extract(year from admit_date) as year, count(*) as count
    from ({all_valid_events_sql})
    group by code, modality, extract(year from admit_date), future_outcome
    """

    codes_by_months_to_outcome_sql = f"""
    select code, modality, future_outcome as outcome, floor(datediff('month', admit_date, outcome_date)) as months_to_outcome, count(*) as count
    from ({all_valid_events_sql})
    group by code, modality,floor(datediff('month', admit_date, outcome_date)), future_outcome
    """

    codes_by_birthyear_sql = f"""
    select code, modality, future_outcome as outcome, extract(year from birthdate) as birthyear,  count(*) as count
    from ({all_valid_events_sql})
    group by code, modality, extract(year from birthdate), future_outcome
    """

    codes_by_age_sql = f"""
    select code, modality, future_outcome as outcome, floor(datediff('year', birthdate, admit_date)) as age, count(*) as count
    from ({all_valid_events_sql})
    group by code, modality, floor(datediff('year', birthdate, admit_date)), future_outcome
    """

    patient_metadata_sql = f"""
        select  pid,
                birthdate, 
                max(last_event_date) as last_event_date,
                min(first_event_date) as first_event_date, 
                sex,
                max(risk_start_date) as risk_start_date,
                min(risk_end_date) as risk_end_date,
                outcome_date,
                future_outcome as outcome,
                region_code,
                split_group,
                max(ydelse_count) as ydelse_count,
                max(prescription_count) as prescription_count,
                max(diag_count) as diag_count,
                max(ydelse_count) + max(prescription_count) + max(diag_count) as total_count,
                max(ydelse_count) + max(prescription_count) + max(diag_count) - {min_events_length} as number_available_trajectories
        from        (select pid,
                            birthdate, 
                            max(admit_date) as last_event_date,
                            min(admit_date) as first_event_date, 
                            sex,
                            greatest(DATE '1995-1-1', birthdate) as risk_start_date,
                            least(outcome_date, birthdate + INTERVAL 30 YEAR) as risk_end_date,
                            outcome_date,
                            future_outcome,
                            region_code,
                            split_group,
                            CASE when modality = 'ydelse' then count(*) else 0 END as ydelse_count,
                            CASE when modality = 'prescription' then count(*) else 0 END as prescription_count,
                            CASE when modality = 'diag' then count(*) else 0 END as diag_count,
                    from ({all_valid_events_sql})
                    group by pid, birthdate, sex, outcome_date, future_outcome, modality, region_code, split_group
                    )
        group by pid, birthdate, sex, outcome_date, future_outcome, region_code, split_group
    """

    patient_diagram_sql = f"""
    select pid, future_outcome, enough_events from patient_metadata
    left join (select pid, 1 as enough_events from ({all_pids_sql})) using (pid)
    where {min_year_admission} - datepart('years', birthdate) =< 40 and {max_year_admission} - date_part('years', birthdate) >= 0
    """

    return (
        all_pids_sql,
        codes_by_year_sql,
        codes_by_months_to_outcome_sql,
        codes_by_birthyear_sql,
        codes_by_age_sql,
        patient_metadata_sql,
        patient_diagram_sql,
    )


def parse_config(config_path):
    config = Namespace(**json.load(open(config_path, "r")))
    return config


def process_duckdb_query(
    conn, sql, pids, value=None, column=None, index=None, do_pivot=True
) -> pd.DataFrame:
    df = conn.execute(sql).df()
    if not do_pivot:
        return df

    if index is None:
        index = df.columns.tolist()
        index.remove(value)
        index.remove(column)

    df = df.pivot(index=index, columns=column, values=value).fillna(0).reset_index()
    return df


def do_counts(config, conn):
    (
        fetch_patients_sql,
        codes_by_year_sql,
        codes_by_months_to_outcome_sql,
        codes_by_birthyear_sql,
        codes_by_age_sql,
        patient_metadata_sql,
        patient_diagram_sql,
    ) = format_metadata_sql(
        min_age_assessment=config.min_age_assessment,
        days_exclusion=config.exclusion_interval * 30,
        min_year_admission=config.min_year_admission,
        max_year_admission=config.max_year_admission,
        min_events_diag=config.min_events_diag,
        min_events_prescription=config.min_events_prescription,
        min_events_ydelse=config.min_events_ydelse,
        min_events_length=config.min_events_length,
    )

    sqls_to_pivot = [
        codes_by_year_sql,
        codes_by_months_to_outcome_sql,
        codes_by_birthyear_sql,
        codes_by_age_sql,
    ]
    columns = ["year", "months_to_outcome", "birthyear", "age", "modality"]
    pids = conn.execute(fetch_patients_sql).df()
    dfs = [
        process_duckdb_query(
            conn=conn, sql=sql, pids=pids, column=column, value="count"
        )
        for sql, column in zip(sqls_to_pivot, columns)
    ]
    dfs.append(
        process_duckdb_query(conn, patient_metadata_sql, pids=pids, do_pivot=False)
    )
    dfs.append(
        process_duckdb_query(conn, patient_diagram_sql, pids=pids, do_pivot=False)
    )
    dfs.append(get_complications_positive_patients(conn, pids, 10))
    return dfs


def generate_counts(run_args, output_dir):
    con = duckdb.connect(run_args.duckdb_database, read_only=True)
    con.sql("PRAGMA enable_progress_bar; SET memory_limit = '500GB'")
    dfs = do_counts(run_args, con)
    con.close()
    pickle.dump(dfs, open(output_dir / "metadata.pkl", "wb"))
    return dfs


####### FUNCTIONS TO SLICE AND MODIFY DATAFRAMES ########


def months_to_year(df: pd.DataFrame):
    df = df.transpose()
    df.index = df.index.to_series().floordiv(12)
    return df.groupby(level=0).sum().transpose()


def slice_codes(codes: pd.Series, modalities: pd.Series, diag_level=3, atc_level=3):
    codes.copy()
    codes.loc[modalities == "diag"] = codes.loc[modalities == "diag"].str[
        : diag_level + 1
    ]
    codes.loc[modalities == "prescription"] = codes.loc[
        modalities == "prescription"
    ].str[: atc_level + 1]
    return codes


def extract_ydelse_chapter(ydelse_code: pd.Series):
    return ydelse_code.str[:2]


def get_years_timespan(start: pd.Series, end: pd.Series):
    return (pd.to_datetime(end) - pd.to_datetime(start)).dt.days / 365.25


###### PRINTING TO CONSOLE #######


def display_diabetes_codes_by_year(months_to_df: pd.DataFrame, num_years=10):
    diabetes_df = months_to_df.loc[
        :, months_to_df.columns.isin(range(-num_years * 3, num_years * 12 + 1))
    ]
    diabetes_df = months_to_year(diabetes_df)
    diabetes_df["code"] = slice_codes(months_to_df["code"], months_to_df["modality"])
    diabetes_df["outcome"] = months_to_df["outcome"]

    diabetes_df = diabetes_df.loc[
        (diabetes_df["code"].str.startswith("DE1"))
        | (diabetes_df["code"].str.startswith("A10"))
    ]
    diabetes_df = diabetes_df.groupby(["code", "outcome"]).sum()
    print("Diabetes codes given around outcome date")
    print(diabetes_df)


def display_count_statistics(patient_df: pd.DataFrame):
    print(f"Number of patients in entire cohort: {len(patient_df['outcome'])}")
    print(
        f"Number of patients with an onset of diabetes: {patient_df['outcome'].sum()}"
    )
    print(f"Number of total events: {patient_df['total_count'].sum(axis=None)}")

    # Median max and min (sd) number of codes used for each modality, positive and negatives
    def _display(df):
        for outcome, data in df.groupby("outcome"):
            type = "Positive" if outcome == 1 else "Negative"
            for modality in [
                "diag_count",
                "prescription_count",
                "ydelse_count",
                "total_count",
            ]:
                print(
                    f"Number of events for {type} {modality}: Min: {data[modality].min()} Median: {data[modality].median()} Max: {data[modality].max()} Sd:{data[modality].std()}, IQR:{data[modality].quantile(0.75) - data[modality].quantile(0.25)} Total: {data[modality].sum()}"
                )

            print(f"Number of patients for {type}: {len(data)}")
            print(f"Number of males for {type}: {(data['sex'] == 1).sum()}")
            print(f"Number of females for {type}: {(data['sex'] == 2).sum()}")
            age = get_years_timespan(data["birthdate"], data["outcome_date"])
            print(
                f"debut age for {type}: Min: {age.min()} Median: {age.median()} Max: {age.max()} Sd: {age.std()} IQR:{age.quantile(0.75) - age.quantile(0.25)} Total: {age.sum()}"
            )

            print(
                f"Number of available trajectories for {type}: Min: {data['number_available_trajectories'].min()} Median: {data['number_available_trajectories'].median()} Max: {data['number_available_trajectories'].max()} Sd:{data['number_available_trajectories'].std()} IQR:{data['number_available_trajectories'].quantile(0.75) - data['number_available_trajectories'].quantile(0.25)} Total: {data['number_available_trajectories'].sum()}"
            )
            data["trajectory_length"] = get_years_timespan(
                data["first_event_date"], data["last_event_date"]
            )
            print(
                f"Trajectory length in year for {type}: Min: {data['trajectory_length'].min()} Median: {data['trajectory_length'].median()} Max: {data['trajectory_length'].max()} Sd:{data['trajectory_length'].std()} IQR:{data['trajectory_length'].quantile(0.75) - data['trajectory_length'].quantile(0.25)} Total: {data['trajectory_length'].sum()}"
            )
            data["trajectory_density"] = data["total_count"] / data["trajectory_length"]
            print(
                f"Events per year {type}: Min: {data['trajectory_density'].min()} Median: {data['trajectory_density'].median()} Max: {data['trajectory_density'].max()} Sd:{data['trajectory_density'].std()} IQR:{data['trajectory_density'].quantile(0.75) - data['trajectory_density'].quantile(0.25)} Total: {data['trajectory_density'].sum()}"
            )

    print("Total")
    _display(patient_df)
    for group, df in patient_df.groupby("split_group"):
        print()
        print(group)
        _display(df)
    for group, df in patient_df.groupby("region_code"):
        print()
        print(group)
        _display(df)


def display_patient_diagram(patients_diagram_df: pd.DataFrame):
    print()
    print("Patient Diagram")
    print(
        f"Total number of patients in denmark younger than 40: {len(patients_diagram_df)}"
    )
    enough_events = patients_diagram_df[patients_diagram_df["enough_events"] == 1]
    print(f"Valid patients with enough events: {len(enough_events)}")
    print(
        f"Valid positive patients with enough events: {(enough_events['future_outcome'] == 1).sum()}"
    )


def display_complications(patients_complications_df: pd.DataFrame):
    print(
        f"Total number of positive patients with complications: {(patients_complications_df['complication_status'] == 0).sum()}"
    )


def display_top_codes_by_modality(age_df: pd.DataFrame, n_codes=5):
    age_df["code"] = slice_codes(age_df["code"], modalities=age_df["modality"])
    tmp = (
        age_df.groupby(["code", "modality", "outcome"])
        .sum()
        .sum(axis=1)
        .reset_index(name="n_events")
    )
    print(f"Top {n_codes} codes for each combination of modality and outcome")
    print(
        tmp.sort_values("n_events", ascending=False)
        .groupby(["modality", "outcome"])
        .head(n_codes)
        .set_index(["modality", "outcome"])
    )


###### PLOTTING TO FILE #######


def barplot_number_of_events_by_modality(patient_df: pd.DataFrame, output_file: Path):
    plt.figure(figsize=(10, 8))
    patient_df[["diag_count", "prescription_count", "ydelse_count", "outcome"]].groupby(
        "outcome"
    ).sum().plot.bar()
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def kde_event_scatter(
    patient_df: pd.DataFrame,
    x: str,
    y: str,
    output_file: Path,
    log_x=True,
    log_y=True,
    xmax=None,
    ymax=None,
):
    x_label = (
        "Number of Prescriptions"
        if x == "prescription_count"
        else "Number of Services"
        if x == "ydelse_count"
        else "Year at risk"
        if x == "trajectory_length"
        else "Number of Total Events"
        if x == "total_count"
        else "Number of Diagnoses"
    )
    y_label = (
        "Number of Prescriptions"
        if y == "prescription_count"
        else "Number of Services"
        if y == "ydelse_count"
        else "Year at risk"
        if y == "trajectory_length"
        else "Number of Total Events"
        if y == "total_count"
        else "Number of Diagnoses"
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    positive = patient_df[patient_df["outcome"] == 1]
    negative = patient_df[patient_df["outcome"] == 0]
    sns.histplot(
        x=x,
        y=y,
        color="grey",
        data=negative,
        ax=axs[0],
        bins=[15, 50],
        log_scale=(log_x, log_y),
        pthresh=0.01,
    )
    sns.histplot(
        x=x,
        y=y,
        color="darkred",
        bins=[15, 50],
        ax=axs[1],
        data=positive,
        log_scale=(log_x, log_y),
        pthresh=0.01,
    )

    def log_formatter(x, pos):
        return f"{int(x):,}" if x >= 1 else f"{x:.2g}"

    for ax in axs.flatten():
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(1, xmax)
        ax.set_ylim(1, ymax)
        ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))

    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def boxplot_events_by_outcome_and_modality(patient_df: pd.DataFrame, output_file: Path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    grouped = patient_df.groupby("outcome")
    grouped[["diag_count", "prescription_count", "ydelse_count"]].boxplot(
        subplots=True, ax=axs
    )

    axs[0].set_ylim((0, 4250))
    axs[1].set_ylim((0, 4250))
    axs[0].set_ylabel("Number of events")
    axs[0].set_title("Positives")
    axs[1].set_title("Negatives")
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def barplot_events_by_birthyear(patient_df: pd.DataFrame, output_file: Path):
    plt.figure(figsize=(12, 8))
    patient_df["birthyear"] = pd.to_datetime(patient_df["birthdate"]).dt.year
    patient_df.loc[
        :, ["birthyear", "diag_count", "prescription_count", "ydelse_count"]
    ].groupby("birthyear").sum().plot.bar(stacked=True)
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def violinplot_age_distribution_by_outcome(patient_df: pd.DataFrame, output_file: Path):
    patient_df["birthyear"] = pd.to_datetime(patient_df["birthdate"]).dt.year
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=patient_df, x="birthyear", y="outcome", orient="h")
    plt.tight_layout()
    plt.ylabel("")
    plt.savefig(fname=output_file, format="pdf")


def histplot_ydelse_counts_by_age_and_type(age_df: pd.DataFrame, output_file: Path):
    cols_to_keep = [col for col in age_df.columns if not isinstance(col, str)]
    ydelse_df = age_df.loc[age_df["modality"] == "ydelse", cols_to_keep]
    ydelse_df["group"] = extract_ydelse_chapter(age_df["code"])
    ydelse_df = ydelse_df.groupby("group").sum()
    ydelse_df = ydelse_df.reset_index().melt(
        id_vars="group", var_name="Age", value_name="Count"
    )
    ydelse_df["Count"] = ydelse_df["Count"] / 1000

    g = sns.FacetGrid(ydelse_df, col="group", col_wrap=6, sharey=False, sharex=False)
    g.map_dataframe(plt.hist, x="Age", weights="Count", color="skyblue")
    g.set_axis_labels("Age", "Count (Thousands)")
    g.set_titles(col_template="{col_name}")
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def lineplot_modality_counts_by_outcome_and_age(
    age_df: pd.DataFrame, output_file, min_age=0, max_age=40
):
    age_df = age_df.groupby(["modality", "outcome"]).mean(numeric_only=True).transpose()
    age_df = age_df.loc[(age_df.index >= min_age) & (age_df.index < max_age)]

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    age_df.xs(1, axis=1, level=1).plot(kind="line", ax=axs[0], title="Positive")
    age_df.xs(0, axis=1, level=1).plot(kind="line", ax=axs[1], title="Negative")
    axs[0].set_ylabel("Average number of events for population in risk")
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def boxplot_trajectory_length_by_outcome(patient_df: pd.DataFrame, output_file: Path):
    patient_df["trajectory_length"] = get_years_timespan(
        patient_df["first_event_date"], patient_df["last_event_date"]
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    patient_df.loc[:, ["trajectory_length", "outcome"]].boxplot(by="outcome")
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def histogram_trajectory_length_by_outcome(patient_df: pd.DataFrame, output_file: Path):
    patient_df["trajectory_length"] = get_years_timespan(
        patient_df["first_event_date"], patient_df["last_event_date"]
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    sns.histplot(
        patient_df.loc[patient_df["outcome"] == 0, "trajectory_length"], ax=axs[0]
    )
    sns.histplot(
        patient_df.loc[patient_df["outcome"] == 1, "trajectory_length"], ax=axs[1]
    )
    axs[0].set_title("Negatives")
    axs[1].set_title("Positives")
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def barplot_modality_counts_by_sex(patient_df: pd.DataFrame, output_file: Path):
    plt.figure(figsize=(10, 8))
    patient_df.loc[
        :, ["sex", "diag_count", "prescription_count", "ydelse_count"]
    ].groupby("sex").sum().plot.bar()
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def histplot_modality_counts_by_trajlength_and_outcome(
    patient_df: pd.DataFrame, output_file: Path
):
    def _doplot(df: pd.DataFrame, output_file: Path, **kwargs):
        fg = sns.FacetGrid(
            df,
            col="modality",
            col_wrap=2,
            sharey=True,
            sharex=True,
            height=4,
            aspect=1.5,
        )
        cax = fg.fig.add_axes([0.82, 0.12, 0.02, 0.8])
        fg.map_dataframe(
            sns.histplot,
            x="trajectory_length",
            y="count",
            stat="count",
            pmax=0.8,
            bins=[10, 35],
            cbar=True,
            cbar_ax=cax,
            data=df,
            **kwargs,
        )
        fg.fig.subplots_adjust(right=0.8)
        fg.set_axis_labels(
            x_var="Year span of trajectory", y_var="Number of events", labelpad=0.5
        )
        plt.savefig(fname=output_file, format="pdf")

    patient_df["trajectory_length"] = get_years_timespan(
        patient_df["first_event_date"], patient_df["last_event_date"]
    )
    patient_df = patient_df[
        [
            "outcome",
            "trajectory_length",
            "ydelse_count",
            "prescription_count",
            "diag_count",
            "total_count",
        ]
    ]
    patient_df = patient_df.melt(
        id_vars=["outcome", "trajectory_length"],
        var_name="modality",
        value_name="count",
    )
    kwargs = [
        {"cmap": "Blues", "binrange": ((0, 30), (0, 2000))},
        {"cmap": "YlOrBr", "binrange": ((0, 30), (0, 2000))},
    ]

    for args, outcome, file_suffix in zip(kwargs, [0, 1], ["negative", "positive"]):
        file = (
            f"{output_file.parent / output_file.stem}_{file_suffix}{output_file.suffix}"
        )
        df = patient_df.loc[patient_df["outcome"] == outcome]
        _doplot(df=df, output_file=file, **args)


def histplot_age_at_diagnosis(patient_df: pd.DataFrame, output_file: Path):
    patient_df = patient_df.loc[patient_df["outcome"] == 1]
    patient_df["age_at_diagnosis"] = get_years_timespan(
        patient_df["birthdate"], patient_df["outcome_date"]
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.histplot(patient_df, x="age_at_diagnosis", kde=True, bins=35)
    plt.tight_layout()
    plt.xlabel("Age at debut")
    plt.ylabel("Number of patients")
    plt.savefig(fname=output_file, format="pdf")


def histplot_last_event_date_by_outcome(patient_df: pd.DataFrame, output_file: Path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    sns.histplot(
        x=pd.to_datetime(patient_df.loc[patient_df["outcome"] == 1, "last_event_date"]),
        kde=True,
        ax=axs[0],
    )
    sns.histplot(
        x=pd.to_datetime(patient_df.loc[patient_df["outcome"] == 0, "last_event_date"]),
        kde=True,
        ax=axs[1],
    )
    plt.tight_layout()
    plt.savefig(fname=output_file, format="pdf")


def describe_metadata(
    year_df: pd.DataFrame,
    month_to_outcome_df: pd.DataFrame,
    birthyear_df: pd.DataFrame,
    age_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    patient_diagram_df: pd.DataFrame,
    patient_complications: pd.DataFrame,
    output_dir: Path,
):
    patient_df["trajectory_length"] = get_years_timespan(
        patient_df["first_event_date"], patient_df["last_event_date"]
    )

    kde_event_scatter(
        patient_df=patient_df,
        x="diag_count",
        y="ydelse_count",
        ymax=2000,
        xmax=1000,
        output_file=output_dir / "event_scatter_diag_ydelse_nolog.pdf",
    )
    kde_event_scatter(
        patient_df=patient_df,
        x="diag_count",
        y="prescription_count",
        ymax=2000,
        xmax=1000,
        output_file=output_dir / "event_scatter_diag_prescription_nolog.pdf",
    )
    kde_event_scatter(
        patient_df=patient_df,
        x="ydelse_count",
        y="prescription_count",
        ymax=2000,
        xmax=2000,
        output_file=output_dir / "event_scatter_ydelse_prescription_nolog.pdf",
    )
    kde_event_scatter(
        patient_df=patient_df,
        x="total_count",
        y="trajectory_length",
        output_file=output_dir / "event_scatter_ydelse_years_nolog.pdf",
        ymax=25,
        xmax=3000,
        log_y=False,
    )
    display_diabetes_codes_by_year(months_to_df=month_to_outcome_df, num_years=10)
    display_count_statistics(patient_df=patient_df)
    display_patient_diagram(patients_diagram_df=patient_diagram_df)
    display_complications(patients_complications_df=patient_complications)

    # What is the most common outcome code
    barplot_number_of_events_by_modality(
        patient_df, output_file=output_dir / "number_of_events.pdf"
    )

    # Distribution of number of codes for positive and negatives
    boxplot_events_by_outcome_and_modality(
        patient_df=patient_df, output_file=output_dir / "boxplot_number_of_events.pdf"
    )

    # Get number of codes by age year
    barplot_events_by_birthyear(
        patient_df=patient_df, output_file=output_dir / "number_of_events_birthyear.pdf"
    )

    # get age distribution
    violinplot_age_distribution_by_outcome(
        patient_df=patient_df, output_file=output_dir / "age_distribution.pdf"
    )

    histplot_ydelse_counts_by_age_and_type(
        age_df=age_df, output_file=output_dir / "ydelse_types.pdf"
    )

    # When do modalities events occur. Age vs Number of events for each modality
    lineplot_modality_counts_by_outcome_and_age(
        age_df=age_df, output_file=output_dir / "age_code_modality.pdf"
    )

    # boxplot of trajectory lengths
    boxplot_trajectory_length_by_outcome(
        patient_df=patient_df, output_file=output_dir / "trajectory_length_boxplot.pdf"
    )

    # Histogram of trajectory lengthts
    histogram_trajectory_length_by_outcome(
        patient_df=patient_df, output_file=output_dir / "trajectory_length.pdf"
    )

    # Get sex information
    barplot_modality_counts_by_sex(
        patient_df=patient_df, output_file=output_dir / "modality_counts_by_sex.pdf"
    )

    # Plot of admission years vs number of codes stratified on outcome
    histplot_modality_counts_by_trajlength_and_outcome(
        patient_df=patient_df, output_file=output_dir / "trajectory_length_binned.pdf"
    )

    # Age at diagnosis
    histplot_age_at_diagnosis(
        patient_df=patient_df, output_file=output_dir / "age_at_diagnosis.pdf"
    )

    # year of end of data
    histplot_last_event_date_by_outcome(
        patient_df=patient_df, output_file=output_dir / "last_event_dates.pdf"
    )


def main():
    args = parse_job_args()
    run_args = parse_config(config_path=os.path.join(args.run_dir, "args.json"))

    OUTPUT_DIR = Path(args.output_dir)
    if not os.path.exists(OUTPUT_DIR / "metadata.pkl") or args.overwrite_metadata:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if args.overwrite_metadata:
            logger.warning(
                f"WARNING: Creating new metadata file at: {OUTPUT_DIR / 'metadata.pkl'}. current metadata data in folder will be overwritten. If this is unwanted, please abort now and restart without --overwrite_metadata",
            )
        dfs = generate_counts(run_args, OUTPUT_DIR)
    else:
        dfs = pickle.load(open(OUTPUT_DIR / "metadata.pkl", "rb"))

    (
        codes_by_year,
        codes_by_month_to_outcome,
        codes_by_birthyear,
        codes_by_age,
        patient_metadata,
        patient_diagram,
        patient_complications,
    ) = dfs

    describe_metadata(
        year_df=codes_by_year,
        month_to_outcome_df=codes_by_month_to_outcome,
        birthyear_df=codes_by_birthyear,
        age_df=codes_by_age,
        patient_df=patient_metadata,
        patient_diagram_df=patient_diagram,
        patient_complications=patient_complications,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
