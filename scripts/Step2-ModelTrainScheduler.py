# This scripts is used to check the environment and existence of necessary files. Python version and package
# dependencies are also checked. If no code map is detected for the json data, one would be generated automatically.
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from os.path import dirname, realpath
from pathlib import Path

sys.path.insert(0, dirname(dirname(realpath(__file__))))
import diabnet.utils.parsing as parsing

CONFIG_NOT_FOUND_MSG = "ERROR! {}: {} config file not found."
RESULTS_PATH_APPEAR_ERR = "ALERT! Existing results for the same config( )."
SUCESSFUL_SEARCH_STR = "Finished! All worker experiments scheduled!"
NO_SCHEDULER_FOUND = "No scheduler found in registry with name {}. Chose between {}"
SCHEDULER_REGISTRY = {}

parser = argparse.ArgumentParser(description="PancNet Grid Search Scheduler.")
parser.add_argument(
    "--search_name", default="untitled-search", type=str, help="The name of the search."
)
parser.add_argument(
    "--experiment_config_path",
    required=True,
    type=str,
    help="Path to the search config.",
)
parser.add_argument(
    "--n_workers", type=int, default=1, help="How many worker nodes to schedule?"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="runs",
    help="The location to store logs and detailCOMMAND_TO_COLLECT_SEARCHed job level result files",
)

parser.add_argument(
    "--scheduler",
    type=str,
    default="torque",
    choices=["torque"],
    help="Which scheduler to use. Choose from ['torque']",
)
parser.add_argument(
    "--group",
    default=None,
    help="unix group associated with the run.",
)
parser.add_argument(
    "--shuffle_experiment_order",
    action="store_true",
    default=False,
    help="Whether to shuffle the order of experiments during grid search.",
)


def RegisterScheduler(scheduler_name):
    def decorator(f):
        SCHEDULER_REGISTRY[scheduler_name] = f
        return f

    return decorator


@RegisterScheduler("torque")
def torque_scheduler(workers):
    """
    Run using moab-torque scheduler. It will run each job within a singularity.
    """

    for worker in workers:
        flag_string = " --save_dir={} --job_file={} ".format(
            args.save_dir,
            os.path.join(args.search_summary_dir, f"{worker}.subexp"),
        )

        shell_cmd = [
            "#!/bin/bash",
            "#PBS -l nodes=1:ppn=12:gpunode:gpus=1",
            "#PBS -l mem=200gb",
            "#PBS -l walltime=08:00:00:00",
            "#PBS -N diabnet",
            f"#PBS -W group_list={args.group}" if args.group is not None else "",
            "#PBS -e /users/people/chrjoh/diabnet.err",
            "#PBS -o /users/people/chrjoh/diabnet.out",
            "module load singularity/3.7.3",
            'SCRATCH_DIR="/local/scratch/$PBS_JOBID"',
            'export SINGULARITY_TMPDIR="$SCRATCH_DIR/singularity/tmp"',
            'export SINGULARITY_CACHEDIR="$SCRATCH_DIR/singularity/cache"',
            "mkdir -p $SCRATCH_DIR",
            "mkdir -p $SINGULARITY_TMPDIR",
            "mkdir -p $SINGULARITY_CACHEDIR",
            'SINGULARITY_NAME="diabnet.latest.sif"',
            'SINGULARITY_PATH="/users/singularity/$SINGULARITY_NAME"',
            f"singularity run -c --nv -B $PBS_O_WORKDIR,/users/projects/diabnet --env CPUS=$(wc -l < $PBS_NODEFILE) --home $HOME --pwd $PBS_O_WORKDIR $SINGULARITY_PATH python $PBS_O_WORKDIR/scripts/worker.py {flag_string} > {args.search_summary_dir}/.$PBS_JOBID.out 2> {args.search_summary_dir}/.$PBS_JOBID.err ",
        ]

        shell_cmd = "\n".join(shell_cmd)
        jobscript = "{}/{}.moab.sh".format(args.search_summary_dir, worker)

        with open(jobscript, "w") as f:
            f.write(shell_cmd)

        print("Launching with moab dispatcher for worker: {}".format(worker))
        subprocess.run(
            ["qsub", jobscript],
            universal_newlines=True,
        )
    return 0


def generate_config_sublist(experiment_config_json):
    job_list = parsing.parse_dispatcher_config(experiment_config_json)

    if args.shuffle_experiment_order:
        random.shuffle(job_list)

    config_sublists = [[] for _ in range(args.n_workers)]
    for k, job in enumerate(job_list):
        config_sublists[k % args.n_workers].append(job)
    workers = [parsing.md5("".join(sublist)) for sublist in config_sublists]

    return job_list, config_sublists, workers


if __name__ == "__main__":
    """
    Dispatch a grid search to one or more machines by creating sub-config files and launch multiple workers.
    """

    args = parser.parse_args()

    assert args.scheduler in SCHEDULER_REGISTRY, NO_SCHEDULER_FOUND.format(
        args.scheduler, list(SCHEDULER_REGISTRY.keys())
    )

    if not os.path.exists(args.experiment_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("master", args.experiment_config_path))
        sys.exit(1)
    experiment_config = json.load(open(args.experiment_config_path, "r"))

    job_list, config_sublists, worker_ids = generate_config_sublist(
        experiment_config_json=experiment_config
    )
    print(
        "Schduling {} dispatchers for {} jobs!".format(
            len(config_sublists), len(job_list)
        )
    )
    [
        print("Sublist {} : {} jobs.".format(worker_ids[i], len(sublist)))
        for i, sublist in enumerate(config_sublists)
    ]
    datestr = datetime.now().strftime("%Y%m%d-%H%M")
    grid_md5 = parsing.md5("".join(job_list))[:8]

    args.save_dir = os.path.join(
        str(Path(args.save_dir).absolute().resolve()),
        f"{args.search_name}_{datestr}_{grid_md5}",
    )
    args.search_summary_dir = os.path.join(args.save_dir, "workers")

    os.makedirs(args.search_summary_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    experiment_summary_path = args.search_summary_dir + "/master.{}.exp".format(
        parsing.md5("".join(job_list))
    )
    if os.path.exists(experiment_summary_path):
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)
    else:
        with open(experiment_summary_path, "w") as out_file:
            out_file.write("worker, job_size\n")
            for i, worker in enumerate(worker_ids):
                out_file.write("{}, {}\n".format(worker, len(config_sublists[i])))

    shutil.copy2(
        args.experiment_config_path,
        os.path.join(args.save_dir, "search_config.json"),
    )

    for i, worker in enumerate(worker_ids):
        with open(
            args.search_summary_dir + "/{}.subexp".format(worker), "w"
        ) as out_file:
            for experiment in config_sublists[i]:
                out_file.write("{}\n".format(experiment))

    SCHEDULER_REGISTRY[args.scheduler](worker_ids)
    print(SUCESSFUL_SEARCH_STR)
    sys.exit(0)
