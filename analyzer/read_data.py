import re
import zipfile
from pathlib import Path
from typing import List, NamedTuple

import pandas as pd

PATH = './data'


class InfoAndData(NamedTuple):
    """csvデータの元号、月、グループ番号、読み出しデータを格納するクラス

    era: 元号
    month: 月
    group: 地域グループ番号
    data: 読み出しデータ
    """
    era: int
    month: int
    group: str
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
    if target in name_l:
        target_zip_idx = name_l.index(target)
    else:
        raise ValueError(f'指定したファイルがないか、ファイル名が間違っています：{target}')
    target_zip_path = path_l[target_zip_idx]
    return target_zip_path


def get_info_and_read_data(filename: str, zip_file: zipfile.ZipFile) -> InfoAndData:
    """zipファイル中の1つのcsvから、元号、月、地域グループ番号を抽出
       データ読み込みを行い、NamedTupleクラスにまとめる

    Args:
        filename (str): csvファイル名(zipfile_name/csvfile_nameの形式)
        zip_file (zipfile.ZipFile): zipfileモジュールで読み出しモードとなっているzipファイルのオブジェクト

    Returns:
        InfoAndData: era->元号、month->月、group->地域グループ番号、data->読み出しデータ(pd.DataFrame)で格納されたNamedTupleクラス
    """
    data_info = re.split(r'(od|\.)', filename)[2]
    group = data_info[-1]
    month = int(data_info[2:-1])
    if month > 12:
        month = int(str(month)[1])
    if month < 4:
        era = int(data_info[:2]) - 1
    else:
        era = int(data_info[:2])
    data = pd.read_csv(zip_file.open(filename), encoding='cp932')
    return InfoAndData(era, month, group, data)


def read_csv_file(target_zip_path: Path) -> List[InfoAndData]:
    """zipファイルからcsvファイルを読み出し、元号、月、地域グループ番号ごとにリストにまとめる

    Args:
        target_zip_path (Path): 調査対象のzipファイルパス

    Returns:
        List[InfoAndData]: 元号、月、地域グループ番号ごとに整理された読み出しデータのリスト
    """
    with zipfile.ZipFile(target_zip_path, 'r') as zip_file:
        data_l = sorted([get_info_and_read_data(filename, zip_file)
                         for filename in zip_file.namelist()])
    return data_l


def read_zip_file(target: str) -> List[InfoAndData]:
    """zipファイルのデータを分類分けして読み込む

    Args:
        target (str): 調査対象のzipファイル名
    """
    data_path = make_data_path()
    target_zip_path = select_zip_file(data_path, target)
    data_l = read_csv_file(target_zip_path)
    return data_l
