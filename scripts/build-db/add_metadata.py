from argparse import ArgumentParser

import duckdb


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--database", action="store", type=str, help="Path to duckdb Database file"
    )
    return parser.parse_args()


def format_sql():
    SQL = f"""
        create or replace table all_pids as (
            select distinct pid from patient_metadata
        );
        
        create or replace table patient_metadata as (
            select pid
                , patient_metadata.sex as sex
                , patient_metadata.birthdate as birthdate
                , split_group
                , case when outcome=1 then 1 else 0 end as future_outcome
                , case when outcome=1 then outcome_registry.outcome_date else patient_metadata.outcome_date end AS outcome_date
                , case when outcome=0 then 0 else 1 end as valid_patient
                from patient_metadata
                left join outcome_registry using (pid)
        );
        create or replace table patient_metadata as (
            select pid
                , CAST(sex as INT1) as sex
                , birthdate
                , split_group
                , CAST(future_outcome as INT1) as future_outcome
                , date_trunc('D', outcome_date) as outcome_date
                , CAST(valid_patient as INT1) as valid_patient
                from patient_metadata
                where valid_patient = 1
        );
        create or replace table dataset as (
            SELECT pid, 
            modality, 
            admit_date, 
            code,
            FROM dataset
            INNER JOIN patient_metadata using (pid)
            WHERE datediff('years', birthdate, admit_date) < 40
                AND valid_patient = 1
            
        );

    """
    return SQL


def main():
    args = parse_args()
    con = duckdb.connect(args.database)
    con.sql(f"PRAGMA threads=32;")
    sql = format_sql()
    con.execute(sql)


if __name__ == "__main__":
    main()
