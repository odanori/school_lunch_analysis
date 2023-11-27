from typing import NamedTuple

import pandas as pd


class InfoAndData(NamedTuple):
    era: int
    month: int
    group: str
    data: pd.DataFrame
