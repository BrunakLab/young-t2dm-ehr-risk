import sys
from argparse import ArgumentParser
from os.path import dirname, realpath

import pandas as pd

sys.path.insert(0, dirname(dirname(dirname(dirname(realpath(__file__))))))

import duckdb

from diabnet.utils.vocab import Vocabulary


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--duckdb_database", help="Database that should be modified")
    parser.add_argument("--table", default="dataset", help="Table to apply the dataset on")
    parser.add_argument("--diag_level", type=int, default=3)
    parser.add_argument("--atc_level", type=int, default=4)
    parser.add_argument("--ydelse_level", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()
    _ = Vocabulary.create_from_duckdb(
        args.duckdb_database, args.diag_level, args.atc_level, args.ydelse_level, table=args.table
    )


if __name__ == "__main__":
    main()
