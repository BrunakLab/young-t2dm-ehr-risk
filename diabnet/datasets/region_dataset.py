import duckdb
import numpy as np
import torch

from diabnet.datasets.patient_history import PatientHistoryDataset
from diabnet.utils.datastructures import EventData, MetaData
from diabnet.utils.sql import query_data_to_memory, query_patient_information

SAMPLED_PATIENTS = []  # hacky way to make sure there is no overlap between train and dev


class RegionalPatientDataset(PatientHistoryDataset):
    """
    Inherits PatientHistoryDataset for all aspects instead of the init.
    We need to fetch patients in a manner that is quite different in comparison.

    Overwrites the _fetch_data method to get patients specific to regions instead of a random split
    """

    def _fetch_data(self, fraction):
        self.duckdb_conn = duckdb.connect(self.args.duckdb_database, read_only=True)
        self.duckdb_conn.sql(
            f"SET threads TO {self.args.num_workers};  PRAGMA enable_progress_bar;"
        )
        metadata = query_patient_information(
            self.args,
            "dataset",
            self.duckdb_conn,
        )
        region_info = metadata[7]
        metadata = metadata[:7]

        if self.split_group == "test":
            metadata = metadata[:, region_info == self.args.test_region]

        else:
            np.random.seed(42)  # These should be sampled the same every run
            metadata = metadata[:, region_info != self.args.test_region]
            if len(SAMPLED_PATIENTS) != 0:
                # We have already loaded a train/dev data set and filter out the sampled patients
                metadata = metadata[:, ~np.isin(metadata[0], SAMPLED_PATIENTS)]
            elif self.split_group == "train":
                # We are loading a train data set and set the patients to filter out from following dev split
                mask = np.random.choice(
                    metadata.shape[1], int(metadata.shape[1] * 0.8), replace=False
                )
                mask.sort()
                metadata = metadata[:, mask]
                SAMPLED_PATIENTS.extend(metadata[0])
            elif self.split_group == "dev":
                mask = np.random.choice(
                    metadata.shape[1], int(metadata.shape[1] * 0.2), replace=False
                )
                mask.sort()
                metadata = metadata[:, mask]
                SAMPLED_PATIENTS.extend(metadata[0])

            else:
                raise NotImplementedError("Patients have not been sampled correctly.")

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
