from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

rcParams['font.family'] = 'MS Gothic'


def output_num_graph(data: pd.DataFrame) -> None:
    """数値データに関するグラフ出力を行う関数
       地域グループごとの給食分量、エネルギーなどを時系列で比較する

    Args:
        data (_type_): preprocessing.pyで前処理されたデータ
    """
    columns = data.columns
    groups = sorted(list(set(data['group'].values)))
    groupby_cols = [g_c for g_c in columns if g_c not in ['era', '献立名', '材料名']]
    contents = [cont for cont in groupby_cols if cont not in ['group', '日付']]
    group_sum_data = data[groupby_cols].groupby(['group', '日付']).sum().reset_index()
    fig, axes = plt.subplots(len(contents), 1, figsize=(20, 30))
    for cont, ax in zip(contents, axes.ravel()):
        for group in groups:
            data_group = group_sum_data[group_sum_data['group'] == group]
            plot_content_graph(data_group['日付'].values, data_group[cont].values, cont, group, ax)
        ax.legend(loc='upper right')
    print(group_sum_data)
    plt.savefig('./result/compare_date_contents_all_group.png')
    plt.close()
    plt.clf()


def plot_content_graph(date: np.ndarray, value: np.ndarray, cont: str, group: str, ax: plt.Axes):
    """地域グループごとに横軸：日付、縦軸：数値データ(給食分量、エネルギーなど)の折れ線グラフを描画

    Args:
        date (np.ndarray): 日付
        value (np.ndarray): 数値データ(給食分量、エネルギー、たんぱく質、脂質、ナトリウムのいずれか)
        cont (str): 数値データカラム名(給食分量、エネルギー、たんぱく質、脂質、ナトリウムのいずれか)
        group (str): 地域グループ番号(a, b, c, d, eのいずれか)
        ax (plt.Axes): グラフ描画エリア
    """
    ax.plot(date, value, label=f'group_{group}')
    ax.set_xlabel('日付')
    ax.set_ylabel(cont)


def make_result_dir() -> None:
    """結果を保存するディレクトリを作成する
    """
    result_path = Path('./result')
    if not result_path.exists():
        result_path.mkdir()


def output_graph(data: pd.DataFrame) -> None:
    """グラフ描画を行う関数

    Args:
        data (pd.DataFrame): preprocessing.pyで前処理したデータ
    """
    make_result_dir()
    output_num_graph(data)
