import zipfile
from pathlib import Path
from typing import List, NamedTuple

import pandas as pd

PATH = './data'


class InfoAndData(NamedTuple):
    era: int
    month: int
    group: int
    data: pd.DataFrame


def make_data_path() -> Path:
    """データを格納したzipファイルまでのパスを取得する
    Returns:
        Path: csvファイルが格納されたzipファイルまでのパス
    """
    data_path = Path(PATH)
    return data_path


def select_zip_file(data_path: Path, target: str) -> Path:
    """調査対象のzipファイルパスを取得する

    Args:
        data_path (Path): csvデータファイルが格納されたzipファイル群までのパス
        target (str): 調査対象のzipファイル名。プログラム実行時に引数として指定

    Returns:
        Path: 調査対象のzipファイルパス
    """
    path_l = [path for path in data_path.iterdir()]
    name_l = [p.name.split('.')[0] for p in path_l]
    target_zip_idx = name_l.index(target)
    target_zip_path = path_l[target_zip_idx]
    return target_zip_path


def get_csv_filename(target_zip_path: Path) -> List[str]:
    """TODO: 編集中

    Args:
        target_zip_path (Path): _description_

    Returns:
        List[str]: _description_
    """
    with zipfile.ZipFile(target_zip_path, 'r') as zip_file:
        n = [filename for filename in zip_file.namelist()]
        print(n)


def read_zip_file(target):
    """zipファイルのデータを分類分けして読み込む

    Args:
        target (str): 調査対象のzipファイル名
    """
    data_path = make_data_path()
    target_zip_path = select_zip_file(data_path, target)
    get_csv_filename(target_zip_path)
