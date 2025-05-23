import numpy as np


class TimeBinner:
    """
    Bins codes into equally spaced bins defined by the number of bins and time.
    NOTE: This can only be used if pad_index for the program is 0, as resize will add zeros to pad the array.
    """

    def __init__(self, start_year, end_year, trajectory_lookback, pad_index, pad_length):
        self._global_bin_space = (end_year - start_year) * 365
        self._global_bin_steps = self._global_bin_space // int(
            trajectory_lookback * 365 / pad_length + 1
        )
        self.bins = np.linspace(0, self._global_bin_space, self._global_bin_steps).round()
        self.pad_index = pad_index
        self.pad_length = pad_length

        if self.pad_index != 0:
            raise ValueError(
                "Pad index can only be 0 for now. It is specified here to ensure that the pad index is actually 0"
            )

    def _dfill(self, a):
        n = a.size
        b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
        return np.arange(n)[b[:-1]].repeat(np.diff(b))

    def get_bins(self):
        padded = np.concatenate([self.bins[:-1], np.full((self.pad_length), self.pad_index)])
        padding_mask = np.concatenate(
            [np.full(self.bins[:-1].shape, False), np.full((self.pad_length), True)]
        )
        return padded, padding_mask

    def _argunsort(self, s):
        n = s.size
        u = np.empty(n, dtype=np.int64)
        u[s] = np.arange(n)
        return u

    def _cumcount(self, a):
        n = a.size
        s = a.argsort()
        i = self._argunsort(s)
        b = a[s]
        return (np.arange(n) - self._dfill(b))[i]

    def _unique_pad(self, a):
        unique_a = np.unique(a)
        padded_a = np.pad(unique_a, (0, a.shape[0] - unique_a.shape[0]))
        return padded_a

    def _unique(self, a):
        """
        Modified version of np.unique, such that it applies over all columns iteratively instead of look for unique columns.
        """
        return np.apply_along_axis(self._unique_pad, 1, a)

    def bin(self, times, codes):
        time_idx = np.digitize(times, self.bins) - 1
        code_idx = self._cumcount(time_idx)

        transformed = np.full((len(self.bins), np.bincount(time_idx).max()), self.pad_index)
        transformed[time_idx, code_idx] = codes
        transformed = transformed[:-1, :]

        transformed_unique = self._unique(transformed)
        transformed_unique = transformed_unique[:, transformed_unique.sum(axis=0) != 0]

        # Add padding on length
        transformed_unique = np.concatenate(
            [
                transformed_unique,
                np.full((self.pad_length, transformed_unique.shape[1]), self.pad_index),
            ]
        )
        return transformed_unique

    def index(self, times):
        """
        Returns the corresponding indexes for bins at that time
        """
        return np.digitize(times, self.bins) - 1
