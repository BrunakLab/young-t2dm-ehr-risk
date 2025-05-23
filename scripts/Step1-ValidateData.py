# This scripts is used to check the environment and existence of necessary files. Python version and package
# dependencies are also checked. If no code map is detected for the json data, one would be generated automatically.
import argparse
import json
import os
import sys
from os.path import dirname, realpath

import duckdb

sys.path.insert(0, dirname(dirname(realpath(__file__))))
import pkg_resources

# Step 1: Check package and update if needed
print("[Step1-CheckFiles][1/3] Checking python environment and version...")

print("[Step1-CheckFiles][1/3] Checking primary location...")
assert "scripts/Step1-CheckFiles.py" not in os.getcwd(), (
    "[Step1-CheckFiles][1/3] Always run your scripts by using under the project root dir, i.e. not under scripts. Abort."
)

print("[Step1-CheckFiles][1/3] Checking package dependencies...")
try:
    pkgs = open("requirements.txt", "r").readlines()
    pkgs = [pkg.rstrip("\b") for pkg in pkgs]
    pkg_resources.require(pkgs)
    print("[Step1-CheckFiles][1/3] All required packages match the desired version.")
except Exception as e:
    print(
        "[Step1-CheckFiles][1/3] One or more packages does not match the desired version. Do `pip install -r "
        "requirements.txt` and then try again."
    )

print("[Step1-CheckFiles][1/3] Checking PancNet core framework...")
sys.path.insert(0, dirname(dirname(realpath(__file__))))
try:
    import diabnet
    from diabnet.utils import parsing
except ModuleNotFoundError:
    print(
        "[Step1-CheckFiles][1/3] PancNet package cannot be found. Check your environment and then continue."
    )


# Step 2: Check experiment configuration and setting yaml.
print("[Step1-CheckFiles][2/3] Checking setting and configuration...")
parser = argparse.ArgumentParser(
    description="Perform a pre-launch test and preprocess data if needed"
)
parser.add_argument(
    "--experiment_config_path",
    required=True,
    type=str,
    help="Path to the search config.",
)

args = parser.parse_args()
try:
    grid_search_config = json.load(open(args.experiment_config_path, "r"))
    print(
        "[Step1-CheckFiles][2/3] Experiment config found at {}.".format(
            args.experiment_config_path
        )
    )
except FileNotFoundError:
    print("[Step1-CheckFiles][2/3] Experiment config not found. Aborting.")
    sys.exit(1)
# Step 3: Check experiment data and vocabulary.
print("[Step1-CheckFiles][3/3] Checking data files...")
duckdb_database_paths = grid_search_config["search_space"]["duckdb_database"]
ydelse_level = grid_search_config["search_space"].get("ydelse_level", [5])[0]
atc_level = grid_search_config["search_space"].get("atc_level", [4])[0]
diag_level = grid_search_config["search_space"].get("diag_level", [3])[0]


for k, duckdb_database_path in enumerate(duckdb_database_paths):
    conn = duckdb.connect(duckdb_database_path)
    idx = "({} out of {})".format(k + 1, len(duckdb_database_paths))
    print(
        "[Step1-CheckFiles][3/3] Checking metadata and associated vocabulary {}...".format(
            idx
        )
    )
    try:
        dataset_columns = conn.execute("select * from dataset limit 1;").df().columns
        metadata_columns = (
            conn.execute("select * from patient_metadata limit 1;").df().columns
        )
    except FileNotFoundError:
        print(
            "[Step1-CheckFiles][3/3]{} Metadata {} with database {} not found. Aborting.".format(
                idx, duckdb_database_path
            )
        )
        sys.exit(1)
    required_metadata_columns = [
        "pid",
        "sex",
        "birthdate",
        "split_group",
        "future_outcome",
        "outcome_date",
    ]
    if not all([col in metadata_columns for col in required_metadata_columns]):
        print("Duckdb metadata table does not have the required columns")
        print("Check that the following columns exist:", required_metadata_columns)

    required_dataset_columns = ["pid", "admit_date", "code_index", "modality_index"]
    if not all([col in dataset_columns for col in required_dataset_columns]):
        print("Duckdb dataset table does not have the required columns")
        print("Check that the following columns exist:", required_dataset_columns)

    database_vocab_config = f"{dirname(duckdb_database_path)}/vocab_config.json"
    RUN_VOCAB_MESSAGE = f"Please run the following command (memory intensive):\npython scripts/build-db/common/create_vocabulary.py --duckdb_database {duckdb_database_path} --diag_level {diag_level} --atc_level {atc_level} --ydelse_level {ydelse_level}"
    if not os.path.exists(database_vocab_config):
        print("Vocabulary has not been applied to data yet.")
        print(RUN_VOCAB_MESSAGE)
        sys.exit(1)

    vocab_config = json.load(open(database_vocab_config, "r"))
    if (
        vocab_config["diag_level"] != diag_level
        or vocab_config["atc_level"] != atc_level
        or vocab_config["ydelse_level"] != ydelse_level
    ):
        print(
            "The arguments specified for diag_level, atc_level or ydelse_level does not match current data"
        )
        print(RUN_VOCAB_MESSAGE)
        sys.exit(1)


print("[Step1-CheckFiles][DONE] All checks passed! Ready to start training.")
