from datetime import datetime

import numpy as np

REFERENCE_DATE = datetime.strptime("1/1/2020", "%d/%m/%Y")


def relative_to_date(arr):
    return np.asarray([REFERENCE_DATE], dtype="datetime64[D]") + np.asarray(
        arr, dtype="timedelta64[D]"
    )
