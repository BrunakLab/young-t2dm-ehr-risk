from argparse import ArgumentParser

import duckdb


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--database", action="store", type=str, help="Path to duckdb Database file"
    )
    parser.add_argument(
        "--outcome-type",
        action="store",
        type=str,
        choices=["all", "T1D", "T2D"],
        help="What outcome type should be created (all, T1D, or T2D",
    )
    return parser.parse_args()


def format_sql(outcome_type):
    SQL = f"""
        CREATE OR REPLACE TABLE outcome_registry as
        with diabetes_clean as (
            SELECT  pid,
                    sex,
                    birthdate,
                    COALESCE(de10_date, CAST('9999-12-31' as DATE)) as de10_date,
                    COALESCE(de11_date, CAST('9999-12-31' as DATE)) as de11_date,
                    COALESCE(other_diag_date, CAST('9999-12-31' as DATE)) as other_diag_date,
                    COALESCE(pcos_date, CAST('9999-12-31' as DATE)) as pcos_date,
                    COALESCE(gdm_date, CAST('9999-12-31' as DATE)) as gdm_date,
                    COALESCE(footterapy_date, CAST('9999-12-31' as DATE)) as footterapy_date,
                    COALESCE(insulin_date, CAST('9999-12-31' as DATE)) as insulin_date,
                    COALESCE(antidiabetic_date, CAST('9999-12-31' as DATE)) as antidiabetic_date,
                    COALESCE(de10_count, 0 ) as de10_count,
                    COALESCE(de11_count, 0) as de11_count,
                    COALESCE(other_diag_count, 0) as other_diag_count,
                    COALESCE(insulin_count, 0) as insulin_count,
                    COALESCE(antidiabetic_count, 0) as antidiabetic_count,
                    COALESCE(footterapy_count, 0) as footterapy_count

            FROM    diabetes_dates
        )
        SELECT *,
                LEAST(de10_date, de11_date, other_diag_date, insulin_date, antidiabetic_date) as outcome_date,
                datediff('years', birthdate, outcome_date) as outcome_age,
                de10_count + de11_count + other_diag_count + footterapy_count as n_diag_codes,
                CASE 
                    WHEN datepart('year', outcome_date) <= 1995 or datepart('year', outcome_date) > 2018 THEN 0
                    WHEN sex = 2 AND outcome_age > 18 AND n_diag_codes + insulin_count = 0 THEN 0
                    WHEN outcome_age > 40 THEN NULL
                    WHEN datediff('days', outcome_date, gdm_date) < 365 OR datediff('days', antidiabetic_date, pcos_date) < 365 THEN 0
                    WHEN n_diag_codes = 0 AND insulin_count + antidiabetic_count < 2 THEN 0
                    ELSE 1
                END as outcome,
                CASE
                    WHEN outcome != 1 THEN NULL
                    WHEN insulin_count = 0 THEN 'T2D'
                    WHEN de10_count > de11_count THEN 'T1D'
                    WHEN datediff('years', birthdate, insulin_date) < 30 OR datediff('years', birthdate, antidiabetic_date) < 15 THEN 'T1D'
                    ELSE 'T2D'
                END as outcome_type,
                CASE
                    WHEN outcome_date = de10_date THEN 'DE10'
                    WHEN outcome_date = de11_date THEN 'DE11'
                    WHEN outcome_date = other_diag_date THEN 'DE12-14'
                    WHEN outcome_date = footterapy_date THEN 'Podiatry'
                    WHEN outcome_date = insulin_date THEN 'Insulin'
                    WHEN outcome_date = antidiabetic_date THEN 'Antidiabetic'
                    ELSE NULL
                END as outcome_reason
        FROM diabetes_clean
        WHERE LEAST(de10_date, de11_date, other_diag_date, footterapy_date, insulin_date, antidiabetic_date) != CAST('9999-12-31' as DATE)
              AND ({"outcome_type = 'T1D' OR outcome_type = 'T2D'" if outcome_type == "all" else f"outcome_type = '{outcome_type}'"} OR outcome = 0)
        ;

    """
    return SQL


def main():
    args = parse_args()
    con = duckdb.connect(args.database)
    con.sql(f"PRAGMA threads=32;")
    sql = format_sql(outcome_type=args.outcome_type)
    con.execute(sql)


if __name__ == "__main__":
    main()
