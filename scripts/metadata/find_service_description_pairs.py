import logging
import math
import multiprocessing as mp
import pickle
import re
import warnings
from argparse import ArgumentParser
from functools import partial

import Levenshtein as lev
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


def checkNumbers(first, second):
    first = [x for x in re.findall(r"(\d+)", first)]
    second = [x for x in re.findall(r"(\d+)", second)]
    return first == second


def reorderWords(first, second, length):
    first = re.sub(r"[.]+", " ", re.sub(r'[?|,|§|:|;|*|=|-|\'|"]+', "", first))
    second = re.sub(r"[.]+", " ", re.sub(r'[?|,|§|:|;|*|=|-|\'|"]+', "", second))

    toReorder = None
    toMatch = None
    if len(first) < len(second):
        toReorder, toMatch = first, second
    else:
        toReorder, toMatch = second, first

    toReorder, toMatch = toReorder.split(" "), toMatch.split(" ")
    toReorder, toMatch = (
        [x.replace(" ", "") for x in toReorder],
        [x.replace(" ", "") for x in toMatch],
    )
    toReorder, toMatch = (
        [x for x in toReorder if x != ""],
        [x for x in toMatch if x != ""],
    )

    ordered = []
    for x in toMatch[: len(toReorder)]:
        if re.sub(r"\D", "", x) != "":
            continue
        best_match = process.extractOne(x, toReorder, scorer=fuzz.token_sort_ratio)
        if (
            best_match[0].replace(".", "") == x[: len(best_match[0].replace(".", ""))]
            and float(len(best_match[0].replace(".", "")) / len(x)) >= length
        ):  # handles abbreviations up to specific length
            ordered.append(x)
        else:
            ordered.append(best_match[0])
        for count, rem in enumerate(toReorder):
            if rem == best_match[0]:
                toReorder.pop(count)
    ordered, toMatch = " ".join(ordered), " ".join(toMatch)
    return ordered.lower(), toMatch.lower()


def process_row(cut, descriptions, threshold, length):
    cut = cut.reset_index(drop=True)
    middle = []
    for outer, x in enumerate(cut.fullcode):
        if (
            len(str(cut["text"][outer]).replace(" ", "")) == 0
            or str(cut["text"][outer]) == "nan"
        ):
            continue
        for inner, y in enumerate(descriptions.fullcode):
            if x == y or str(descriptions["text"][inner]) == "nan":
                continue

            first, second = (
                str(cut["text"][outer]).lower(),
                str(descriptions["text"][inner]).lower(),
            )
            firstFilt, secondFilt = reorderWords(
                str(cut["text"][outer]), str(descriptions["text"][inner]), length=length
            )

            pad = min(len(firstFilt), len(secondFilt))  # pad the longer string
            if pad == 0:
                continue

            # pair descriptions together if they are similar with OR without filtering
            if (
                lev.ratio(firstFilt[:pad], secondFilt[:pad]) > threshold
                and len(firstFilt) > 0
                and len(secondFilt) > 0
            ) or (
                lev.ratio(first[:pad], second[:pad]) > threshold
                and len(first) > 0
                and len(second) > 0
            ):
                middle.append([cut["text"][outer], descriptions["text"][inner]])

    return middle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", help="Raw documentation file")
    parser.add_argument(
        "--output", help="Pairs of similar descriptions to manually look through"
    )
    parser.add_argument(
        "--abb-length",
        help="Fraction of string that can be abbrivation",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--levenshtein-threshold",
        help="Levenshtein ratio required to classify as potential pair",
        type=float,
        default=0.35,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    workers = mp.cpu_count()

    descriptions = pd.read_csv(args.input, encoding="utf-8", dtype=str)
    desc_pairs = []
    pool = mp.Pool(workers)
    process_func = partial(
        process_row,
        descriptions=descriptions,
        threshold=args.levenshtein_threshold,
        length=args.abb_length,
    )
    for count, res in enumerate(
        pool.map(
            process_func,
            [index for index in np.array_split(descriptions, int(workers))],
        )
    ):
        desc_pairs.append(res)
    desc_pairs = [
        y for x in desc_pairs for y in x
    ]  # 3d -> 2d, a list of pairs of descriptions that are similar
    toRem = []
    for count, x in enumerate(desc_pairs):
        if (
            x[0] == x[1] or not checkNumbers(x[0], x[1])
        ):  # If any pair of description differs in their numbers remove them, fx: "1. konsultation" and "2. konsultation"
            toRem.append(count)
    for i in sorted(toRem, reverse=True):
        del desc_pairs[i]

    with open(args.output, "wb") as f:
        pickle.dump(desc_pairs, f)


if __name__ == "__main__":
    main()
