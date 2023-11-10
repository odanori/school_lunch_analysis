import argparse
import json
from pathlib import Path

import pandas as pd

import analyzer.output as output
import analyzer.read_data as read_data
from analyzer.config import Config
from analyzer.preprocessing import Preprocessing


def make_parser():
    """実行時に指定する引数を受け付ける

    Returns:
        args: 受け付けた引数
    """
    parser = argparse.ArgumentParser(
        description='調べたいデータのzipファイル名や前処理方法を記載したconfig.jsonファイルを指定')
    parser.add_argument('config_json_file', help='設定jsonファイル名')
    args = parser.parse_args()
    return args


def load_settings(config_json_name: str) -> Config:
    """調査対象zipファイルの指定や、前処理条件の設定jsonファイルの読み込み

    Args:
        config_json_name (str): 設定ファイル名

    Returns:
        Dict: 設定項目と設定値が辞書型で整理されたもの
              例: {ayalysis_target: "kyushoku"
                   preprocess: {
                        rename_col: "日付"
                      }
                   }
    """
    config_path = Path('./settings') / config_json_name
    print(config_path)
    config_json = json.load(open(config_path, encoding='utf-8_sig'))
    config = Config(config_json)
    return config


def prepare_data(config: Config) -> pd.DataFrame:
    """データの読み出しと前処理を行う関数

    Args:
        config (Config): 読み出し対象や前処理を行うカラムの設定

    Returns:
        pd.DataFrame: 前処理が行われた各データが1つに結合されたもの
    """
    info_and_data_l = read_data.read_zip_file(config.analysis_target)

    preprocess = Preprocessing(config)
    processed_data_l = [preprocess.data_preprocess(
        info_and_data.data, info_and_data.era, info_and_data.group) for info_and_data in info_and_data_l]
    concat_processed_data = pd.concat(processed_data_l, axis=0)
    return concat_processed_data


def output_data(data: pd.DataFrame) -> None:
    """結果を出力する関数
       現在はグラフ出力のみ

    Args:
        data (pd.DataFrame): prepare_data関数で処理されたデータ
    """
    output.output_graph(data)


def run():
    """データ読み込みから前処理、グラフ化までの実行関数
    """
    args = make_parser()
    config = load_settings(args.config_json_file)
    data = prepare_data(config)
    output_data(data)


if __name__ == '__main__':
    run()
