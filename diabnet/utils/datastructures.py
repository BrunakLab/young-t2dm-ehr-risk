from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from diabnet.utils.date import relative_to_date


class MetaData:
    """
    Convenience for grapping the metadata
    """

    def __init__(self, dataset: torch.Tensor) -> None:
        self._dataset = dataset

    def __getitem__(self, idx):
        """
        For subsetting into batches
        """
        return MetaData(self._dataset[:, idx])

    @property
    def pid(self):
        return self._dataset[0].numpy().astype(int)

    @cached_property
    def birthdate(self):
        return relative_to_date(self._dataset[1].numpy())

    @property
    def future_outcome(self):
        return self._dataset[2].numpy().astype(int)

    @cached_property
    def outcome_date(self):
        return relative_to_date(self._dataset[3].numpy())

    @cached_property
    def min_date(self):
        return relative_to_date(self._dataset[4].numpy())

    @cached_property
    def max_date(self):
        return relative_to_date(self._dataset[5].numpy())

    @cached_property
    def postive_endpoint_date(self):
        return relative_to_date(self._dataset[6].numpy())

    @property
    def min_idx(self):
        return self._dataset[7].numpy().astype(int)

    @property
    def max_idx(self):
        return self._dataset[8].numpy().astype(int)

    def __repr__(self) -> str:
        return self._dataset.__repr__()

    def __len__(self):
        return len(self.pid)

    @property
    def shape(self):
        return self._dataset.shape


class EventData:
    """
    Convenience wrapper around the events to make subsetting easier.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def fetch_patient(self, min_idxs, max_idxs):
        sorting = self._dataset[min_idxs : max_idxs + 1, 0].argsort()
        return PatientEvents(
            pids=self._dataset[min_idxs : max_idxs + 1, 3][sorting],
            code_tokens=self._dataset[min_idxs : max_idxs + 1, 1][sorting],
            modality_tokens=self._dataset[min_idxs : max_idxs + 1, 2][sorting],
            admit_dates=relative_to_date(self._dataset[min_idxs : max_idxs + 1, 0][sorting]),
        )

    def __repr__(self) -> str:
        return self._dataset.__repr__()

    def __len__(self):
        return len(self._dataset)

    @property
    def shape(self):
        return self._dataset.shape


@dataclass
class PatientEvents:
    pids: torch.Tensor
    code_tokens: torch.Tensor
    modality_tokens: torch.Tensor
    admit_dates: np.ndarray


class MatchData:
    """
    Wrapper to fetch matched patient indexes that can be used to sample from in metadata

    """

    def __init__(
        self, negative_pids: torch.Tensor, min_idxs: torch.Tensor, max_idxs: torch.Tensor
    ) -> None:
        # The min/max idxs have the structure age, decade, idx
        self.pids = negative_pids
        self.min_idxs = min_idxs
        self.max_idxs = max_idxs

    def sample_patient(self, age_at_assessement: int, year_of_assessment: int, n):
        random_idx = None
        decade = year_of_assessment // 10
        min_mask = (self.min_idxs[:, 0] == age_at_assessement) & (self.min_idxs[:, 1] == decade)
        max_mask = (self.max_idxs[:, 0] == age_at_assessement) & (self.max_idxs[:, 1] == decade)

        while min_mask.sum() == 0 and max_mask.sum() == 0:
            print(
                f"Could not find patient for age: {age_at_assessement}. Increasing age by 1 and rechecking"
            )
            age_at_assessement += 1
            min_mask = (self.min_idxs[:, 0] == age_at_assessement) & (
                self.min_idxs[:, 1] == decade
            )
            max_mask = (self.max_idxs[:, 0] == age_at_assessement) & (
                self.max_idxs[:, 1] == decade
            )

            if age_at_assessement > 120:
                print("Selecting Random")
                random_idx = torch.randint(len(self.min_idxs), (1,)).item()

        if random_idx is None:
            min_idxs = self.min_idxs[min_mask, 2].item()
            max_idxs = self.max_idxs[max_mask, 2].item()
        else:
            min_idxs = self.min_idxs[int(random_idx), 2].item()
            max_idxs = self.max_idxs[int(random_idx), 2].item()

        pid_idxs = torch.randint(int(min_idxs), int(max_idxs), (n,))

        return self.pids[pid_idxs]
