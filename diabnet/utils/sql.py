PROSPECTIVE_DATE_CUTOFF = None
from typing import Tuple

import numpy as np
import pandas as pd

from diabnet.utils.date import REFERENCE_DATE


def query_patient_information(args, split_group, connection):
    # Filter of dates are atleast 3 events (1 from each modality)... as we take the max of the minimum dates
    days_exclusion = args.exclusion_interval * 30
    if args.exclude_codes:
        exclusion_codes = open(args.exclude_codes, "r").read().splitlines()
    else:
        exclusion_codes = [""]

    query_valid_dates = f"""
        with joined_dataset as (
            select pid, modality, admit_date, birthdate, future_outcome, outcome_date
            from dataset
            inner join (select pid, birthdate, future_outcome, outcome_date from patient_metadata where split_group = ?) using (pid)
            
            where datediff('days', birthdate, admit_date)/365 > {args.min_age_assessment}
            and datediff('days', admit_date, outcome_date)>{days_exclusion}
            and ((future_outcome and admit_date<outcome_date) or (not future_outcome and datediff('days', admit_date, outcome_date)>{args.negative_lookahead}*365))
            and extract(year from admit_date) between {args.min_year_admission} and {args.max_year_admission}
            AND code NOT IN ({",".join("?" for _ in range(len(exclusion_codes)))})
        )
    
        select pid, max(min_date) as min_date, max(max_date) as max_date
        from (select pid, modality, min(admit_date) as min_date, max(admit_date) as max_date
              from joined_dataset 
              group by pid, modality
              having (modality = 'diag' and count(*) >= {args.min_events_diag})
                or   (modality = 'prescription' and count(*) >= {args.min_events_prescription})
                or   (modality = 'ydelse' and count(*) >= {args.min_events_ydelse})
              )
        group by pid
        having count(*) = 3
        order by pid

    """
    valid_dates = connection.execute(
        query_valid_dates, [split_group] + exclusion_codes
    ).df()

    valid_dates = valid_dates.set_index("pid")
    # Split is defined by region if test_region is set, so need all splits
    if hasattr(args, "test_region"):
        patient_metadata = (
            connection.execute(
                f"select pid, sex, birthdate, outcome_date, future_outcome, region_code from patient_metadata and sex in ({','.join(['?' for _ in args.sex])})",
                args.sex,
            )
            .df()
            .set_index("pid")
        )

    else:
        patient_metadata = (
            connection.execute(
                f"select pid, sex, birthdate, outcome_date, future_outcome from patient_metadata where split_group = ? and sex in ({','.join(['?' for _ in args.sex])})",
                [split_group] + args.sex,
            )
            .df()
            .set_index("pid")
        )
    patient_metadata = patient_metadata.join(valid_dates, how="inner").reset_index()
    patient_metadata = patient_metadata.sort_values("pid")

    # filter on number of unique codes
    min_number_unique_codes = f"""
    select pid
    from (
            select distinct pid, code_index, modality_index 
            from dataset
            inner join (SELECT * FROM patient_metadata WHERE split_group = ?) using (pid)
            where extract(year from admit_date) between {args.min_year_admission} and {args.max_year_admission}
              AND admit_date < outcome_date
         )
    group by pid
    having count(*) >= {args.min_unique_events}
    """
    patients_min_unique_events = connection.execute(
        min_number_unique_codes, [split_group]
    ).df()
    patient_metadata = patient_metadata[
        patient_metadata["pid"].isin(patients_min_unique_events["pid"])
    ]
    # Filter that people have at least X years of data
    patient_metadata = patient_metadata[
        patient_metadata["max_date"]
        >= patient_metadata["min_date"] + pd.DateOffset(years=args.min_years_history)
    ]
    assert patient_metadata["future_outcome"].sum() > 0, (
        "Not fetching any positive patient from the DB - check your filter query"
    )
    patient_metadata = patient_metadata.sort_values("pid")

    # Set the minimum date for sampling a positive trajectory
    positive_mask = patient_metadata["future_outcome"] == 1
    patient_metadata["positive_endpoint_date"] = patient_metadata["min_date"]
    potential_positive_dates = patient_metadata.loc[
        positive_mask, "outcome_date"
    ] - pd.DateOffset(months=max(args.month_endpoints))
    potential_positive_dates = pd.concat(
        [
            patient_metadata.loc[positive_mask, "positive_endpoint_date"],
            potential_positive_dates,
        ],
        axis=1,
    ).max(axis=1)
    patient_metadata.loc[positive_mask, "positive_endpoint_date"] = pd.concat(
        [patient_metadata.loc[positive_mask, "max_date"], potential_positive_dates],
        axis=1,
    ).min(axis=1)

    # Change dates to a duration for storing in tensor
    patient_metadata["min_date_to_ref"] = (
        patient_metadata["min_date"] - REFERENCE_DATE
    ).dt.days
    patient_metadata["max_date_to_ref"] = (
        patient_metadata["max_date"] - REFERENCE_DATE
    ).dt.days
    patient_metadata["positive_endpoint_date_to_ref"] = (
        patient_metadata["positive_endpoint_date"] - REFERENCE_DATE
    ).dt.days
    patient_metadata["birthdate_to_ref"] = (
        patient_metadata["birthdate"] - REFERENCE_DATE
    ).dt.days
    patient_metadata["outcome_to_ref"] = (
        patient_metadata["outcome_date"] - REFERENCE_DATE
    ).dt.days

    result = [
        patient_metadata["pid"].values,
        patient_metadata["birthdate_to_ref"].values,
        patient_metadata["future_outcome"].values,
        patient_metadata["outcome_to_ref"].values,
        patient_metadata["min_date_to_ref"].values,
        patient_metadata["max_date_to_ref"].values,
        patient_metadata["positive_endpoint_date_to_ref"].values,
    ]
    if hasattr(args, "test_region"):
        result.append(patient_metadata["region_code"].values)
    return np.stack(result)


def query_data_to_memory(
    conn, args, pids
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if args.exclude_codes:
        exclusion_codes = open(args.exclude_codes, "r").read().splitlines()
    else:
        exclusion_codes = [""]
    data = conn.execute(
        f"""SELECT pid, datediff('days',?,admit_date) as admit_date, code_index, modality_index 
                from dataset
                where modality IN ({",".join(["?" for _ in args.modalities])})
                AND extract(year from admit_date) between {args.min_year_admission} and {args.max_year_admission}
                AND pid IN ({",".join("?" for _ in range(len(pids)))})
                AND code NOT IN ({",".join("?" for _ in range(len(exclusion_codes)))})
                ORDER BY pid

            """,
        [REFERENCE_DATE] + args.modalities + pids.tolist() + exclusion_codes,
    ).df()
    data.loc[data["code_index"].isna(), "code_index"] = 1
    data["code_index"] = data["code_index"].astype(int)
    data.reset_index(inplace=True, names="index")
    indexes = data.groupby("pid")["index"].agg(["max", "min"])
    indexes.rename(
        columns={"max": "event_end_idx", "min": "event_start_idx"}, inplace=True
    )

    return (
        indexes["event_start_idx"].values,
        indexes["event_end_idx"].values,
        data[["admit_date", "code_index", "modality_index", "pid"]],
    )
