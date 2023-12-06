from typing import List

import pandas as pd
from sqlalchemy import create_engine, inspect

from v2_analyzer.settings.config import Config


def make_engine():
    """DB接続のためのエンジンを作成
    """
    engine = create_engine(Config.POSTGRES_URI)
    return engine


def dispose_engine(engine):
    """DB接続のためのエンジンを閉じる
    """
    engine.dispose()


def get_table_names(engine) -> List[str]:
    """DBのテーブル名を取得する。ダウンロードしたファイル名を管理するテーブルは除く

    Args:
        engine : DB接続のエンジン

    Returns:
        List[str]: データを保存したテーブル名のリスト
    """
    inspector = inspect(engine)
    all_table_names = inspector.get_table_names()
    # 取得ファイル名管理テーブルを除外する
    table_names = [name for name in all_table_names if name != Config.POSTGRES_FILENAME_TABLE]
    return table_names


def fetch_data_from_db(table_name: str, engine) -> pd.DataFrame:
    """DBからデータを取得する

    Args:
        table_name (str): データ取得対象のテーブル名
        engine : DB接続のエンジン

    Returns:
        pd.DataFrame: 対象のテーブルから取得したデータ
    """
    query = f'SELECT * FROM {table_name}'
    data = pd.read_sql_query(query, engine)
    return data


def sort_data(data: pd.DataFrame) -> pd.DataFrame:
    """元号、地域グループ番号、日付順にデータを並べ替える

    Args:
        data (pd.DataFrame): DBから取得したデータ

    Returns:
        pd.DataFrame: 元号、地域グループ番号、日付順に並べ替えられたデータ
    """
    sorted_data = data.sort_values(
        by=['era', 'area_group', 'date'], ascending=[True, True, True]).reset_index(drop=True)
    return sorted_data


def take_data():
    """DBからデータを取得する
       全データを取得し、1つのテーブルにまとめる

    Returns:
        _type_: _description_
    """
    engine = make_engine()
    table_names = get_table_names(engine)

    data_list = [fetch_data_from_db(table_name, engine) for table_name in table_names]
    concat_data = pd.concat(data_list, axis=0)
    all_data = sort_data(concat_data)

    dispose_engine(engine)
    del data_list
    del concat_data
    return all_data
