import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd


class InfoAndData(NamedTuple):
    era: int
    month: int
    group: str
    data: pd.DataFrame


def get_info_and_read_data(filepath: Path) -> InfoAndData:
    filename = filepath.name
    data_info = re.split(r'od|\.', filename)[-2]
    group = data_info[-1]
    month = int(data_info[2:-1])
    if month > 12:
        month = int(str(month)[1])
    if month < 4:
        era = int(data_info[:2]) - 1
    else:
        era = int(data_info[:2])
    data = pd.read_csv(filepath, encoding='cp932')
    return InfoAndData(era, month, group, data)


def read_data(file_path: str) -> InfoAndData:
    data_info = get_info_and_read_data(file_path)
    return data_info
