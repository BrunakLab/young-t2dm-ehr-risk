import random
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.utils import data

from diabnet.utils.datastructures import EventData, MetaData, PatientEvents
from diabnet.utils.sql import query_data_to_memory, query_patient_information
from diabnet.utils.time_binner import TimeBinner
from diabnet.utils.vocab import PAD_CODE, CodeParser, Vocabulary

MAX_TIME_EMBED_PERIOD_IN_DAYS = 120 * 365
MIN_TIME_EMBED_PERIOD_IN_DAYS = 10

SUMMARY_MSG = (
    "Constructed disease progression {} dataset with {} records from {} patients, "
    "and the following class balance:\n  {}"
)


def load_required_indexes(path, vocabulary):
    codes = open(path, "r").read().splitlines()
    indexes = {}
    for code_modality in codes:
        code, modality = code_modality.split("\t")
        index = vocabulary.code_to_index(code, modality)
        if modality not in indexes:
            indexes[modality] = [index]
        else:
            indexes[modality].append(index)
    return indexes


class PatientHistoryDataset(data.Dataset):
    def __init__(
        self,
        args,
        split_group,
        vocabulary: Vocabulary,
        fraction: Optional[float] = None,
    ):
        """
            Dataset for survival analysis based on categorical disease history information.

        Args:
            args : The global arguments (see utils/parsing.py)
            split_group (str): Use any of ['train', 'test', 'dev']"
            vocabulary: The vocabulary to map between tokens and codes.
            fraction: Subset negatives to a fraction of the available negatives. This is mostly useful for attributions

        Returns:
            torch.utils.data.Dataset

        """
        super(PatientHistoryDataset, self).__init__()
        self.args = args
        self.split_group = split_group
        self.PAD_CODE = PAD_CODE
        self.vocabulary = vocabulary

        self.required_tokens = (
            load_required_indexes(args.codes_required_in_trajectory, vocabulary)
            if args.codes_required_in_trajectory
            else None
        )

        self.trajectories_per_patient = (
            self.args.max_train_indices
            if split_group == "train"
            else self.args.max_eval_indices
        )
        # Lookback period for single trajectory is half of overall lookback
        self.binner = TimeBinner(
            start_year=args.min_year_admission,
            end_year=args.max_year_admission,
            trajectory_lookback=args.trajectory_lookback,
            pad_length=self.args.pad_size,
            pad_index=self.vocabulary.code_to_index(
                self.PAD_CODE, "diag"
            ),  # modality does not matter
        )
        print(
            f"DuckDB working on {self.args.duckdb_database} using {self.args.num_workers} threads"
        )
        self._fetch_data(fraction)

    def _fetch_data(self, fraction):
        self.duckdb_conn = duckdb.connect(self.args.duckdb_database, read_only=True)
        self.duckdb_conn.sql(
            f"SET threads TO {self.args.num_workers};  PRAGMA enable_progress_bar;"
        )
        metadata = query_patient_information(
            self.args, self.split_group, self.duckdb_conn
        )

        if fraction:
            metadata = metadata[
                :, (metadata[2] == 1) | (np.mod(metadata[0], int(1 / fraction)) == 0)
            ]
        elif fraction == 0:
            metadata = metadata[:, metadata[2] == 1]

        if self.args.patients_exclusion_file and self.split_group == "train":
            patients_to_filter = pd.read_csv(self.args.patients_exclusion_file)
            metadata = metadata[:, ~np.isin(metadata[0], patients_to_filter["pid"])]

        if self.args.patients_inclusion_file and self.split_group == "train":
            patients_to_include = pd.read_csv(self.args.patients_inclusion_file)
            metadata = metadata[:, np.isin(metadata[0], patients_to_include["pid"])]

        print(f"Fetching data to memory")
        event_start, event_end, event_data = query_data_to_memory(
            conn=self.duckdb_conn,
            args=self.args,
            pids=metadata[0],
        )
        self.duckdb_conn.close()

        metadata = np.concatenate(
            [
                metadata,
                np.expand_dims(event_start, axis=0),
                np.expand_dims(event_end, axis=0),
            ]
        )
        self.metadata = MetaData(torch.from_numpy(metadata))

        self.event_data = EventData(torch.from_numpy(event_data.values))
        print(
            f"Num positive patients in {self.split_group} are {self.metadata.future_outcome.sum()}"
        )
        print(
            f"Num patients in {self.split_group} are {len(self.metadata.future_outcome)}"
        )

    def get_trajectories(
        self,
        trajectory_indexes,
        event_data: PatientEvents,
        metadata: MetaData,
    ):
        """
        Given a patient, multiple trajectories can be extracted by sampling partial histories.
        """
        # Get tokens up to the max index
        # Some patients are not valid in cases when subsetting data for debugging
        if len(trajectory_indexes) == 0:
            return []
        max_index = max(trajectory_indexes)
        code_tokens = event_data.code_tokens[: max_index + 1]
        relative_admit_dates = (
            event_data.admit_dates[max_index] - event_data.admit_dates[: max_index + 1]
        ).astype(int)
        relative_admit_dates = torch.from_numpy(relative_admit_dates)
        modality_tokens = event_data.modality_tokens[: max_index + 1]

        trajectory_bin_indexes = self.binner.index(
            relative_admit_dates[trajectory_indexes]
        )
        # for each modality bin and store result

        binned_tokens = {}
        for modality in self.args.modalities:
            modality_mask = modality_tokens == self.vocabulary.modality_to_index(
                modality
            )

            binned_tokens[modality] = self.binner.bin(
                relative_admit_dates[modality_mask], code_tokens[modality_mask]
            )

        all_bins, all_padding = self.binner.get_bins()
        trajectories = []

        for idx, bin_idx in zip(trajectory_indexes, trajectory_bin_indexes):
            if self.required_tokens:
                seen_required_index = False
                for modality in self.args.modalities:
                    modality_mask = (
                        modality_tokens == self.vocabulary.modality_to_index(modality)
                    )
                    if np.isin(
                        code_tokens[modality_mask], self.required_tokens[modality]
                    ).any():
                        seen_required_index = True
                        break
            if not seen_required_index:
                continue

            final_admission_date = event_data.admit_dates[idx]
            age_at_final = (final_admission_date - metadata.birthdate).astype(int)

            trajectory = {
                "patient_id": metadata.pid,
                "tokens": code_tokens.unsqueeze(0),
                "future_outcome": metadata.future_outcome,
                "admit_date": str(final_admission_date),
                "trajectory_time_length": (
                    final_admission_date.astype("datetime64[D]")
                    - event_data.admit_dates[0].astype("datetime64[D]")
                ).astype(np.int32),
                "age": age_at_final,
                "year": final_admission_date.astype("datetime64[Y]").astype(int)
                + 1970,  # unix epoch,
            }
            # readjust the bins and ignore the padding
            bins = all_bins[bin_idx : bin_idx + self.args.pad_size].copy()
            padding = all_padding[bin_idx : bin_idx + self.args.pad_size]

            bins[~padding] = bins[~padding] - bins[0]
            ages = np.where(padding, bins, age_at_final + bins)
            trajectory.update(
                {"time_seq": bins, "age_seq": ages, "padding_seq": padding}
            )

            for modality in self.args.modalities:
                trajectory.update(
                    {
                        f"{modality}_tokens": binned_tokens[modality][
                            bin_idx : bin_idx + self.args.pad_size
                        ],
                    }
                )

            y, y_seq, y_mask, time_at_event, days_to_censor = self.get_label(
                future_outcome=metadata.future_outcome,
                outcome_date=metadata.outcome_date,
                final_admission_date=final_admission_date,
            )

            trajectory.update(
                {
                    "y": y,
                    "y_seq": y_seq,
                    "y_mask": y_mask,
                    "time_at_event": time_at_event,
                    "days_to_censor": days_to_censor,
                    "exam": "",
                }
            )
            trajectories.append(trajectory)
        return trajectories

    def get_label(self, future_outcome, outcome_date, final_admission_date):
        """
        Args:
            patient (dict): The patient dictionary which includes all the processed diagnosis events.
            until_idx (int): Specify the end point for the partial trajectory.

        Returns:
            outcome_date: date of pancreatic cancer diagnosis for cases (cancer patients) or
                          END_OF_TIME_DATE for controls (normal patients)
            time_at_event: the position in time vector (default: [3,6,12,36,60]) which specify the outcome_date
            y_seq: Used as golds in cumulative_probability_layer
                   An all zero array unless ever_develops_panc_cancer then y_seq[time_at_event:]=1
            y_mask: how many years left in the disease window
                    ([1] for 0:time_at_event years and [0] for the rest)
                    (without linear interpolation, y_mask looks like complement of y_seq)

            Ex1:  A partial disease trajectory that includes pancreatic cancer diagnosis between 6-12 months
                  after time of assessment.
                    time_at_event: 2
                    y_seq: [0, 0, 1, 1, 1]
                    y_mask: [1, 1, 1, 0, 0]
            Ex2:  A partial disease trajectory from a patient who never gets pancreatic cancer diagnosis
                  but died between 36-60 months after time of assessment.
                    time_at_event: 1
                    y_seq: [0, 0, 0, 0, 0]
                    y_mask: [1, 1, 1, 1, 0]
        """

        days_to_censor = (outcome_date - final_admission_date) / np.timedelta64(1, "D")
        num_time_steps, max_time = (
            len(self.args.month_endpoints),
            max(self.args.month_endpoints),
        )
        y = int(days_to_censor < (max_time * 30) and future_outcome)
        y_seq = np.zeros(num_time_steps)
        if days_to_censor < (max_time * 30):
            time_at_event = min(
                [
                    i
                    for i, mo in enumerate(self.args.month_endpoints)
                    if days_to_censor < (mo * 30)
                ]
            )
        else:
            time_at_event = num_time_steps - 1

        if y:
            y_seq[time_at_event:] = 1
        y_mask = np.array(
            [1] * (time_at_event + 1) + [0] * (num_time_steps - (time_at_event + 1))
        )

        assert time_at_event >= 0 and len(y_seq) == len(y_mask)
        return (
            y,
            y_seq.astype("float64"),
            y_mask.astype("float64"),
            time_at_event,
            days_to_censor,
        )

    def __len__(self):
        return len(self.metadata)

    def adjust_min_date(self, future_outcome, first_positive_endpoint_date, min_date):
        choose_positive_traj = np.random.choice(
            [0, 1],
            size=len(min_date),
            p=[
                1 - self.args.positive_trajectory_fraction,
                self.args.positive_trajectory_fraction,
            ],
        )
        return np.where(
            (future_outcome == 1) & (choose_positive_traj == 1),
            first_positive_endpoint_date,
            min_date,
        )

    def extract_trajectory_indexes(
        self, min_date, max_date, admit_dates, n_samples=None
    ):
        if n_samples is None:
            n_samples = self.trajectories_per_patient
        months_diff = (max_date - admit_dates) / np.timedelta64(1, "D") / 30
        search_space = (max_date - min_date) / np.timedelta64(1, "D") / 30
        chosen_months = np.arange(0, search_space, step=1, dtype=np.int32)
        # Index of last event in trajectory at least m month from outcome (aka length of that subtrajectory)
        months_indexes = [(months_diff >= m).sum() - 1 for m in chosen_months]
        valid_indexes = [
            idx for idx in set(months_indexes) if idx + 1 >= self.args.min_events_length
        ]
        valid_indexes = sorted(valid_indexes)

        if self.split_group == "train":
            return random.choices(valid_indexes, k=min(n_samples, len(valid_indexes)))

        return valid_indexes[-n_samples:]

    def __getitem__(self, idxs):
        metadata = self.metadata[idxs]
        # Adjust trajectory space for half positives (random) to only contain positive trajectories
        if self.split_group == "train":
            min_dates = self.adjust_min_date(
                metadata.future_outcome,
                metadata.postive_endpoint_date,
                metadata.min_date,
            )
        else:
            min_dates = metadata.min_date
        trajectories = []
        for i in range(len(metadata.pid)):
            event_data = self.event_data.fetch_patient(
                metadata.min_idx[i], metadata.max_idx[i]
            )
            trajectory_indexes = self.extract_trajectory_indexes(
                min_date=min_dates[i],
                max_date=metadata.max_date[i],
                admit_dates=event_data.admit_dates,
            )
            # debug queries of data sometimes have no valid indexes
            trajectories.extend(
                self.get_trajectories(
                    trajectory_indexes=trajectory_indexes,
                    metadata=metadata[i],
                    event_data=event_data,
                )
            )
        return trajectories
