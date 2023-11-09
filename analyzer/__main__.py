import argparse
import json
from pathlib import Path
from typing import Dict

import analyzer.read_data as read_data
from analyzer.config import Config


def make_parser():
    parser = argparse.ArgumentParser(
        description='調べたいデータのzipファイル名や前処理方法を記載したconfig.jsonファイルを指定')
    parser.add_argument('config_json_file', help='設定jsonファイル名')
    args = parser.parse_args()
    return args


def load_settings(config_json_name: str) -> Dict:
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
    config_json = json.load(open(config_path))
    config = Config(config_json)
    return config


def run():
    """データ読み込みから前処理、グラフ化までの実行関数
    """
    args = make_parser()
    config = load_settings(args.config_json_file)
    data_l = read_data.read_zip_file(config.analysis_target)
    data = data_l[0].data
    preprocessing.preprocess(data)


if __name__ == '__main__':
    run()
