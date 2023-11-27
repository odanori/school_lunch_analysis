import re
from pathlib import Path

import pandas as pd
from preparation.data_info import InfoAndData


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
    data = add_info(data, era, group)
    return InfoAndData(era, month, group, data)


def add_info(data: pd.DataFrame, era: int, group: str) -> pd.DataFrame:
    data.insert(0, 'era', era)
    data.insert(1, 'group', group)
    return data


def read_data(file_path: Path) -> InfoAndData:
    data_info = get_info_and_read_data(file_path)
    return data_info
