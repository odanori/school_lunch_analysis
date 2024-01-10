import datetime
import itertools
import math
import re
from datetime import date
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import fftpack
from scipy.signal import argrelextrema
from scipy.stats import kstest, mannwhitneyu, norm, ttest_ind
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller

GRP = 'e'
EX_COL = ['era', 'area_group', 'date', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
AGG_D = {
    'amount_g': 'sum',
    'carolies_kcal': 'sum',
    'protein_g': 'sum',
    'fat_g': 'sum',
    'sodium_mg': 'sum'
}
TARGET_L = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']


def output_num_graph(data: pd.DataFrame) -> None:
    """各種数値データ(分量やカロリーなど)の日付ごとの合算値を地域グループごとに比較するグラフ作成
       横軸が日付、縦軸が各種数値データ
       数値データ-> amount_g: 分量(g), carolies_kcal: エネルギー(kcal), protein_g: たんぱく質(g)
                   fat_g: 脂質(g), sodium_mg: ナトリウム(mg)

    Args:
        data (pd.DataFrame): DBから取り出したデータ
    """
    area_groups = sorted(list(set(data['area_group'].values)))

    extract_cols = ['area_group', 'date', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    contents = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']

    colors = px.colors.qualitative.Plotly

    sum_contents_by_area = data[extract_cols].groupby(['area_group', 'date']).sum().reset_index()
    num_graph = len(contents)

    fig = make_subplots(rows=num_graph, cols=1)
    for i, cont in enumerate(contents):
        row = i + 1
        col = 1
        graph_num = i + 1
        for e, area in enumerate(area_groups):
            data_group = sum_contents_by_area[sum_contents_by_area['area_group'] == area]
            color = colors[e]
            plot_content(data_group, cont, area, row, col, color, fig)
        layout(cont, graph_num, fig)

    fig.show()


def plot_content(data_group: pd.DataFrame, content: str, area: str, row: int, col: int, color: str, fig) -> None:
    """横軸が日付、縦軸が数値データの折れ線グラフを描画する

    Args:
        data_group (pd.DataFrame): 地域グループごとに抽出されたデータ
        content (str): 数値データ名(例: fat_g->脂質(g)など)
        area (str): 地域グループ名(例: a->地域グループa)
        row (int): グラフエリアの行位置番号
        col (int): グラフエリアの列位置番号
        color (str): 折れ線グラフの色
        fig : グラフエリアのオブジェクト
    """
    date = data_group['date'].values
    cont_value = data_group[content].values
    trace = go.Scatter(
        x=date, y=cont_value, mode='lines', name=f'{content}_group_{area}', line=dict(color=color), showlegend=True
        )
    fig.add_trace(trace, row=row, col=col)


def layout(content: str, graph_num: int, fig):
    """グラフエリアごとにレイアウトを設定する
       x軸の目盛間隔、y軸名、凡例位置、グラフの大きさなど

    Args:
        content (str): 数値データ名(例: fat_g->脂質(g)など)
        graph_num (int): グラフエリア識別番号。各グラフエリアごとに軸名や目盛間隔を指定する
                         例: graph_num1, graph_num2でグラフエリア1、グラフエリア2それぞれのレイアウト設定
        fig : グラフエリアのオブジェクト
    """
    fig.update_layout(**{f'xaxis{graph_num}': dict(dtick='M1', title='date')},
                      **{f'yaxis{graph_num}': dict(title=content)},
                      legend=dict(xanchor='left'),
                      width=1500,
                      height=1500,
                      )


def output_graph(data: pd.DataFrame) -> None:
    # output_num_graph(data)
    # plot_scatter_graph(data)
    # plot_correlogram(data)
    # plot_matrix(data)
    # plot_hist(data)
    # plot_qq(data)
    # kolmogorov_smirnov_test(data)
    # confirm_identity_of_menu(data)
    # whelch_t_test(data)
    # mann_whiteney_u_test(data)
    # check_stationariness(data)
    # manipulate_unsteady_and_plot_correlogram(data)
    # check_significance_fft_signal(data)
    # manual_fft(data)
    # menu_kinds_analysis(data)
    # menu_value_analysis(data)
    menu_contents_analysis(data)


def plot_scatter_graph(data) -> None:
    cp_data = data[['era', 'area_group', 'date', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']].copy()
    ex_data_4_a = cp_data[
        (cp_data['era'] == 4) & (cp_data['area_group'] == 'e') & (cp_data['date'] <= datetime.date(2022, 12, 31))]
    ex_data_5_a = cp_data[
        (cp_data['era'] == 5) & (cp_data['area_group'] == 'e') & (cp_data['date'] <= datetime.date(2023, 12, 31))]
    ex_data_4_a_grp = ex_data_4_a.groupby(['date']).agg({
        'area_group': 'count', 'amount_g': 'sum',
        'carolies_kcal': 'sum', 'protein_g': 'sum',
        'fat_g': 'sum', 'sodium_mg': 'sum'}).reset_index()
    ex_data_5_a_grp = ex_data_5_a.groupby(['date']).agg({
        'area_group': 'count', 'amount_g': 'sum',
        'carolies_kcal': 'sum', 'protein_g': 'sum',
        'fat_g': 'sum', 'sodium_mg': 'sum'}).reset_index()
    ex_data_4_a_grp = ex_data_4_a_grp.rename(columns={'area_group': 'count_ingredients'})
    ex_data_5_a_grp = ex_data_5_a_grp.rename(columns={'area_group': 'count_ingredients'})
    # ex_data_4_a_grp['week'] = ex_data_4_a_grp['date'].apply(lambda x: x.strftime("%a"))
    # ex_data_5_a_grp['week'] = ex_data_5_a_grp['date'].apply(lambda x: x.strftime("%a"))

    custom_order = range(4, 13)
    month_days_4_a = ex_data_4_a_grp['date'].apply(lambda x: x.month).value_counts().reindex(custom_order)
    month_days_5_a = ex_data_5_a_grp['date'].apply(lambda x: x.month).value_counts().reindex(custom_order)
    days_diff = month_days_4_a - month_days_5_a
    print(month_days_4_a)
    print(month_days_5_a)
    print(days_diff)

    print(len(ex_data_4_a_grp), len(ex_data_5_a_grp))
    print(ex_data_4_a_grp)
    print(ex_data_5_a_grp)
    ex_data_4_a_grp['month'] = ex_data_4_a_grp['date'].apply(lambda x: x.month)
    ex_data_5_a_grp['month'] = ex_data_5_a_grp['date'].apply(lambda x: x.month)

    fig = make_subplots(rows=5, cols=1)
    colors = px.colors.qualitative.Light24
    contents = ['count_ingredients', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    for cont in range(5):
        row = cont + 1
        col = 1
        cont_name = contents[cont+1]
        ct_ingres = []
        cont_values = []
        for i in custom_order:
            color = colors[i]
            diff = ex_data_4_a_grp[ex_data_4_a_grp['month'] == i].reset_index(drop=True) - ex_data_5_a_grp[ex_data_5_a_grp['month'] == i].reset_index(drop=True)
            # trace = go.Scatter(x=diff['count_ingredients'].values, y=diff[cont_name].values, mode='markers', name=f'{i}_{cont_name}', line=dict(color=color), showlegend=True)
            # fig.add_trace(trace, row=row, col=col)
            ct_ingres += list(diff['count_ingredients'].values)
            cont_values += list(diff[cont_name].values)
            # print(diff)
        ct_ingres = [val for val in ct_ingres if not np.isnan(val)]
        cont_values = [val for val in cont_values if not np.isnan(val)]
        r = calc_correlation_coeff(ct_ingres, cont_values)
        print(f'{cont_name}, r={r}')
        trace = go.Scatter(x=ct_ingres, y=cont_values, mode='markers', name=f'{cont_name}', line=dict(color=color), showlegend=True)
        fig.add_trace(trace, row=row, col=col)
        fig.update_layout(**{f'xaxis{row}': dict(title='diff_count_ingredients r4 - r5')},
                          **{f'yaxis{row}': dict(title=cont_name)},
                          legend=dict(xanchor='left'),
                          width=1500,
                          height=4000,)
    fig.show()


def calc_correlation_coeff(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_times_sum = np.sum([i**2 for i in x])
    y_times_sum = np.sum([i**2 for i in y])
    sx = np.sqrt((x_times_sum / n) - x_mean**2)
    sy = np.sqrt((y_times_sum / n) - y_mean**2)
    sxy = np.sum([(i - x_mean) * (k - y_mean) for i, k in zip(x, y)]) / n

    r = sxy / (sx * sy)
    return r


def plot_correlogram(data):
    grp = 'e'
    cont = 'sodium_mg'
    cp_data = data[['era', 'area_group', 'date', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']].copy()
    ex_data_4 = cp_data[
        (cp_data['era'] == 4) & (cp_data['area_group'] == grp) & (cp_data['date'] <= datetime.date(2022, 12, 31))]
    ex_data_5 = cp_data[
        (cp_data['era'] == 5) & (cp_data['area_group'] == grp) & (cp_data['date'] <= datetime.date(2023, 12, 31))]
    ex_data_4_grp = ex_data_4.groupby(['date']).agg({
        'area_group': 'count', 'amount_g': 'sum',
        'carolies_kcal': 'sum', 'protein_g': 'sum',
        'fat_g': 'sum', 'sodium_mg': 'sum'}).reset_index()
    ex_data_5_grp = ex_data_5.groupby(['date']).agg({
        'area_group': 'count', 'amount_g': 'sum',
        'carolies_kcal': 'sum', 'protein_g': 'sum',
        'fat_g': 'sum', 'sodium_mg': 'sum'}).reset_index()
    ex_data_4_grp = ex_data_4_grp.rename(columns={'area_group': 'count_ingredients'})
    ex_data_5_grp = ex_data_5_grp.rename(columns={'area_group': 'count_ingredients'})
    ex_data_4_grp['month'] = ex_data_4_grp['date'].apply(lambda x: x.month)
    ex_data_5_grp['month'] = ex_data_5_grp['date'].apply(lambda x: x.month)

    correlo_cols = ['month'] + [cont]
    tmp_df = ex_data_4_grp[correlo_cols]
    print(ex_data_4_grp)
    fig = make_subplots(rows=9, cols=1)
    for i in range(4, 13):
        row = i - 3
        col = 1
        auto_corr_l = []
        ex_values = tmp_df[(tmp_df['month'] == i)][cont].values
        days = 10  # len(ex_values)
        for k in range(days):
            # h = d + 1
            # ex_df.loc[:, f'{correlo_cols[1]}_h{h}'] = ex_df[correlo_cols[1]].shift(h)
            auto_corr = calc_auto_correlation(ex_values, k)
            auto_corr_l.append(auto_corr)
        print(ex_values)
        print(auto_corr_l)
        # raise Exception()
        trace = go.Bar(x=list(range(len(auto_corr_l))), y=auto_corr_l, name=f'month:{i}_grp:{grp}', showlegend=True)
        fig.add_trace(trace, row=row, col=col)
        fig.update_layout(**{f'yaxis{row}': dict(title=cont, range=[-0.6, 1])},
                          **{f'xaxis{row}': dict(title='lag(days)', dtick=1, range=[-0.5, 22])},
                          width=1500,
                          height=3000)
    fig.show()

    # ex_data_4_grp['amount_g_h1'] = ex_data_4_grp['amount_g'].shift(1)
    # print(ex_data_4_grp)
    # print(ex_data_5_grp)


def calc_auto_correlation(data, k):
    t = len(data)
    mean_d = np.mean(data)
    cov = np.sum([(data[i] - mean_d) ** 2 for i in range(t)]) / t
    auto_cov = np.sum([(data[i] - mean_d) * (data[i - k] - mean_d) for i in range(k, t)]) / t
    auto_corr = auto_cov / cov
    return auto_corr


def plot_matrix(data):
    grp = 'b'
    cp_data = data.copy()
    ex_data = cp_data[['era', 'area_group', 'date', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']]
    ex_data_4 = ex_data[(ex_data['era'] == 4) & (ex_data['area_group'] == grp) & (ex_data['date'] <= datetime.date(2022, 12, 31))]
    ex_data_5 = ex_data[(ex_data['era'] == 5) & (ex_data['area_group'] == grp) & (ex_data['date'] <= datetime.date(2023, 12, 31))]

    ex_data_4_grp = ex_data_4.groupby(['date']).agg({
        'area_group': 'count',
        'amount_g': 'sum',
        'carolies_kcal': 'sum',
        'protein_g': 'sum',
        'fat_g': 'sum',
        'sodium_mg': 'sum'
    }).reset_index()
    ex_data_5_grp = ex_data_5.groupby(['date']).agg({
        'area_group': 'count',
        'amount_g': 'sum',
        'carolies_kcal': 'sum',
        'protein_g': 'sum',
        'fat_g': 'sum',
        'sodium_mg': 'sum'
    }).reset_index()

    ex_data_4_grp = ex_data_4_grp.rename(columns={'area_group': 'count_ingredients'})
    ex_data_5_grp = ex_data_5_grp.rename(columns={'area_group': 'count_ingredients'})
    ex_data_4_grp['month'] = ex_data_4_grp['date'].apply(lambda x: x.month)
    ex_data_5_grp['month'] = ex_data_5_grp['date'].apply(lambda x: x.month)
    # colors = px.colors.qualitative.Light24

    # 各種数値データ同士の相関係数を見ている
    cols = ['count_ingredients', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    corr_cols = list(itertools.combinations(cols, 2))
    # point_color = {i: colors[i-4] for i in range(4, 13)}
    corr_4_5 = []
    for d in [ex_data_4_grp, ex_data_5_grp]:
        df = d.copy()
        # df['point_color'] = ex_data_4_grp['month'].map(point_color)
        df['month'] = df['month'].astype('str')
        fig = px.scatter_matrix(df, dimensions=cols, color='month', title='r')
        fig.update_layout(width=2000,
                          height=2000)
        print(df)
        corr_l = []
        for cols in corr_cols:
            x = df[cols[0]].values
            y = df[cols[1]].values
            r = calc_correlation_coeff(x, y)
            corr_l.append(r)
        corr_4_5.append(corr_l)
        del df

        # print(corr_l)
    corr_df = pd.DataFrame({'cols': [col for col in corr_cols],
                            'corr_4': corr_4_5[0],
                            'corr_5': corr_4_5[1]})
    print(f'group_{grp}')
    print(corr_df)

    # fig.show()


def plot_hist(data):

    ex_col = ['era', 'area_group', 'date', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    agg_d = {
        #'area_group': 'count',
        'amount_g': 'sum',
        'carolies_kcal': 'sum',
        'protein_g': 'sum',
        'fat_g': 'sum',
        'sodium_mg': 'sum'
    }
    cp_data = data.copy()
    ex_data = cp_data[ex_col]

    target_l = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    xrange_l = [[400, 750], [550, 750], [18, 40], [12, 30], [300, 1600]]
    yrange_l = [[0, 20], [0, 20], [0, 20], [0, 20], [0, 25]]
    for target, xrange, yrange in zip(target_l, xrange_l, yrange_l):
        for e in [4, 5]:
            if e == 4:
                w_c = 2022
            else:
                w_c = 2023
            ex_data_wc = ex_data[(ex_data['era'] == e) & (ex_data['date'] <= datetime.date(w_c, 12, 31))]
            ex_data_wc_grp = ex_data_wc.groupby(['date', 'area_group']).agg(agg_d).reset_index()
            ex_data_wc_grp = ex_data_wc_grp.rename({'area_group': 'count_ingredients'})
            ex_data_wc_grp['month'] = ex_data_wc_grp['date'].apply(lambda x: str(x.month))
            print(ex_data_wc_grp)
            fig = px.histogram(ex_data_wc_grp, x=target, color='area_group', barmode='overlay', marginal='box', nbins=40)
            fig.update_layout(xaxis=dict(range=xrange),
                              yaxis=dict(range=yrange),
                              title=f'r{e}_{target}')
            fig.show()
    del data


def calc_prob(x: int, n: int) -> float:
    """データ順位から累積密度関数(確率)を出す
        順位の累積確率の位置

    Args:
        x (int): データ順位
        n (int): データ総数

    Returns:
        float: 累積密度関数(確率)
    """
    # p = (x - 0.5) / n
    p = x / (n + 1)
    return p

def calc_expc(x: float, n: int, m: float, s: float) -> float:
    """標準正規分布の逆関数を求め、平均と標準偏差から期待値を出す

    Args:
        x (float): 累積分布関数(確率)
        n (int): データ数
        m (float): データの平均値
        s (float): データの標準偏差

    Returns:
        float: 期待値
    """
    # 分位数の確認
    inverse_norm = np.percentile(np.random.normal(size=n), 100 * x)
    expc = m + (s * inverse_norm)
    return expc


def calc_cumulative_distribution(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """累積分布関数を求め、標本の平均と標準偏差から期待値を求める

    Args:
        data (pd.DataFrame): 全種類の数値データが格納されたDataFrame
        target (str): 対象数値データのカラム名

    Returns:
        pd.DataFrame: 数値データの順位、累積分布関数の値、期待値が計算された対象数値データ
    """
    _ex = data[target].reset_index().drop(columns='index')
    n = len(_ex)
    _ex['rank'] = _ex[target].rank(method='min')
    _ex['prob'] = _ex['rank'].apply(calc_prob, args=[n])
    _ex_mean = np.mean(_ex[target].values)
    _ex_std = (sum([(i - _ex_mean) ** 2 for i in _ex[target].values]) / n) ** 0.5
    _ex['expc'] = _ex['prob'].apply(calc_expc, args=[n, _ex_mean, _ex_std])
    print(_ex_mean, _ex_std)
    return _ex

def plot_qq(data):
    np.random.seed(200)

    # norm = np.random.normal(size=146)
    # # plt.hist(norm)
    # plt.boxplot(norm, vert=False)
    # # 標準正規分布にて、x パーセンタイル点での値を見る
    # percentile = [10, 25, 30, 75, 100]
    # print(np.percentile(norm, percentile))
    # plt.show()
    # raise Exception()

    # メニューに地域差がない可能性。グループdを例にとる
    grp = 'd'
    ex_col = ['era', 'area_group', 'date', 'amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    agg_d = {
        'amount_g': 'sum',
        'carolies_kcal': 'sum',
        'protein_g': 'sum',
        'fat_g': 'sum',
        'sodium_mg': 'sum'
    }
    cp_data = data.copy()
    ex_data = cp_data[ex_col]

    target_l = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']

    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data_wc = ex_data[(ex_data['era'] == e) & (ex_data['date'] <= datetime.date(w_c, 12, 31))]
        ex_data_wc_grp = ex_data_wc.groupby(['date', 'area_group']).agg(agg_d).reset_index()
        ex_data_wc_grp = ex_data_wc_grp[ex_data_wc_grp['area_group'] == grp]

        fig = make_subplots(rows=3, cols=2, subplot_titles=target_l)
        for i, target in enumerate(target_l):
            row = i // 2 + 1
            col = i % 2 + 1
            # _ex = ex_data_wc_grp[target].reset_index().drop(columns='index')
            # n = len(_ex)
            # _ex['rank'] = _ex[target].rank(method='min')
            # _ex['prob'] = _ex['rank'].apply(calc_prob, args=[n])
            # _ex_mean = np.mean(_ex[target].values)
            # _ex_std = (sum([(i - _ex_mean) ** 2 for i in _ex[target].values]) / len(_ex[target])) ** 0.5
            # _ex['expc'] = _ex['prob'].apply(calc_expc, args=[n, _ex_mean, _ex_std])
            _ex = calc_cumulative_distribution(ex_data_wc_grp, target)
            line = go.Scatter(x=_ex[target].values, y=_ex[target].values, mode='lines', line=dict(color='red'), name=f'{target}_group_{grp}_ideal', showlegend=False)
            trace = go.Scatter(x=_ex[target].values, y=_ex['expc'].values, mode='markers', name=f'{target}_group_{grp}', showlegend=False)
            fig.add_trace(trace, row=row, col=col)
            fig.add_trace(line, row=row, col=col)
            fig.update_layout(**{f'xaxis{row}': dict(title='実測値')},
                              **{f'yaxis{row}': dict(title='期待値')},
                            #   **{f'title{row}': dict(title=f'{target}')},
                              width=1000,
                              height=2000,
                              )

            # print(_ex_mean, _ex_std)
            print(_ex)
        fig.show()

        # plt.scatter(_ex['target'].values, _ex['expc'].values)
        # plt.plot(sorted(_ex['expc'].values), sorted(_ex['expc'].values), color='red')
        # plt.show()


def standarization(x: float, m: float, s: float):
    z = (x - m) / s
    return z


def standard_normal_cdf(x):
    return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0


def kolmogorov_smirnov_test(data):
    np.random.seed(200)
    cp_data = data.copy()
    ex_data = cp_data[EX_COL]

    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data_wc = ex_data[(ex_data['era'] == e) & (ex_data['date'] <= datetime.date(w_c, 12, 31))]
        ex_data_wc_grp = ex_data_wc.groupby(['date', 'area_group']).agg(AGG_D).reset_index()
        ex_data_wc_grp = ex_data_wc_grp[ex_data_wc_grp['area_group'] == GRP]

        for target in TARGET_L:
            _ex = ex_data_wc_grp[target].reset_index().drop(columns='index')
            n = len(_ex)
            _ex['rank'] = _ex[target].rank(method='min')
            _ex['prob'] = _ex['rank'].apply(calc_prob, args=[n])
            _ex_mean = np.mean(_ex[target].values)
            _ex_std = (sum([(i - _ex_mean) ** 2 for i in _ex[target].values]) / (n - 1)) ** 0.5
            _ex[f'{target}_sd'] = _ex[target].apply(standarization, args=[_ex_mean, _ex_std])
            _ex = _ex.sort_values('rank')
            _norm = np.random.normal(size=n)
            norm_df = pd.DataFrame({'norm': _norm})
            # norm_df['cdf'] = norm_df['norm'].apply(standard_normal_cdf)
            norm_df['rank'] = norm_df['norm'].rank(method='min')
            norm_df['prob'] = norm_df['rank'].apply(calc_prob, args=[n])
            norm_df = norm_df.sort_values('rank')
            # d_statistic = np.max(np.abs(_ex[f'{target}_sd'].values - norm_df['norm'].values))
            # d_statistic = np.max(np.abs(_ex['prob'].values - norm_df['prob'].values)) * 2
            # ks = d_statistic * (n*n / (n + n)) ** 0.5
            ks_statistic, p_value = kstest(_ex[f'{target}_sd'].values, 'norm', alternative='two-sided')
            # print(_ex)
            # print(norm_df)
            # print(d_statistic)
            print(f'scipy, kstestの値, 統計量:{ks_statistic}, p値{p_value}')
            # print(ks)
            # print(kstest(_ex[f'{target}_sd'].values, 'norm'))
            # plt.plot(_ex[f'{target}_sd'].values, _ex['prob'].values)
            # plt.plot(norm_df['norm'].values, norm_df['prob'].values)
            # plt.show()


            # gptが教えたCDFの差を自分で計算する方法
            # p値の求め方は確率密度関数の積分の結果のような形か？
            _ex['theo_cdf'] = norm.cdf(_ex[f'{target}_sd'].values)
            d_st = np.max(np.abs(_ex['theo_cdf'].values - np.arange(1, len(_ex['theo_cdf']) + 1) / len(_ex['theo_cdf'])))
            p_value_ather = 2 * np.exp(-2 * d_st ** 2 * len(_ex['theo_cdf']))
            # print(_ex)
            print(f'累積分布関数cdfの差を取った結果の値, 統計量:{d_st}, p値:{p_value_ather}')
            # aplha: 有意水準, 両側検定、5%に設定(片側2.5%)
            # 帰無仮説H0の下でp値がalphaよりも小さいと、H0はめったに起こらない(棄却域に入る)として、H0を棄却する
            # p値がalphaよりも大きいと、H0が間違っているとは言えない(棄却しない。H0が正しいとも言えない)
            alpha = 0.025
            if p_value > alpha:
                print(f'帰無仮説を棄却しない。r{e}_{target}の分布は標準正規分布と異なるとは言えない')
            else:
                print(f'帰無仮説を棄却する。r{e}_{target}の分布は標準正規分布と異なると言える')


def confirm_identity_of_menu(data):
    cp_data = data.copy()
    cp_data['month'] = cp_data['date'].apply(lambda x: x.month)
    for month in range(4, 13):
        ex_data = cp_data[cp_data['month'] == month]
        menu_l = []
        for r in [4, 5]:
            ex_data_grp = ex_data[(ex_data['area_group'] == GRP) & (ex_data['era'] == r)]
            menu = list(set(ex_data_grp['menu'].values))
            menu_l.append(menu)
        if set(menu_l[0]) == set(menu_l[1]):
            print(f'r4, r5の{month}月のmenuは同一')
        else:
            print(f'r4, r5の{month}月のmenuは異なる')
            r4_only = [m for m in menu_l[0] if m not in menu_l[1]]
            r5_only = [m for m in menu_l[1] if m not in menu_l[0]]
            print(r4_only, r5_only)
            # result = np.setdiff1d(menu_l[0], menu_l[1])
            # print(result)


def whelch_t_test(data):
    """H0: 2群間の平均値に差がない
       H1: 2群間の平均値に差がある
       有意水準alpha:5%, 両側検定(0.025)
       p > alpha -> H0を棄却できない、2群間の平均値が異なるとは言えない
       p < alpha -> H0を棄却する、2群間の平均値は異なると言える(何かしら意味がある、有意である)

    Args:
        data (_type_): _description_
    """
    val_col = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    cp_data = data.copy()
    cp_data = cp_data[EX_COL]
    ex_data_4 = cp_data[(cp_data['era'] == 4) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(2022, 12, 31))]
    ex_data_5 = cp_data[(cp_data['era'] == 5) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(2023, 12, 31))]
    ex_data_4_agg = ex_data_4.groupby(['date']).agg(AGG_D).reset_index()
    ex_data_5_agg = ex_data_5.groupby(['date']).agg(AGG_D).reset_index()

    n_4 = len(ex_data_4_agg)
    n_5 = len(ex_data_5_agg)
    print(n_4, n_5)

    # 4~12月合算でのr4、r5の平均値に差があるかを確かめる
    ex_4 = ex_data_4_agg[val_col].T.values
    ex_5 = ex_data_5_agg[val_col].T.values
    era_diff = list(map(lambda x, y: ttest_ind(x, y, equal_var=False), ex_4, ex_5))
    era_diff_df = pd.DataFrame(era_diff)
    era_diff_df['p < 0.050'] = era_diff_df['pvalue'].apply(lambda x: True if x < 0.025 else False)
    era_diff_df['d'] = era_diff_df['statistic'].apply(lambda x: abs(x) * ((n_4 + n_5) / (n_4 * n_5)) ** 0.5)
    era_diff_df.insert(0, 'name', val_col)
    print(era_diff)
    print(era_diff_df)


def mann_whiteney_u_test(data):
    """各月ごとの各数値データに差があるかを調べる
       各月のデータはr4とr5を合算してn < 60 のため
    Args:
        data (_type_): _description_
    """
    val_col = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    cp_data = data.copy()
    cp_data = cp_data[EX_COL]
    ex_data_4 = cp_data[(cp_data['era'] == 4) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(2022, 12, 31))]
    ex_data_5 = cp_data[(cp_data['era'] == 5) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(2023, 12, 31))]
    ex_data_4_agg = ex_data_4.groupby(['date']).agg(AGG_D).reset_index()
    ex_data_5_agg = ex_data_5.groupby(['date']).agg(AGG_D).reset_index()

    ex_data_4_agg['month'] = ex_data_4_agg['date'].apply(lambda x: x.month)
    ex_data_5_agg['month'] = ex_data_5_agg['date'].apply(lambda x: x.month)

    def calc_r(u, n1, n2):
        n = n1 + n2
        nu = (n1 * n2) / 2
        std = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5
        z = abs(u - nu) / std
        r = z / (n ** 0.5)
        return r

    for m in range(4, 13):
        n1 = len(ex_data_4_agg[ex_data_4_agg['month'] == m])
        n2 = len(ex_data_5_agg[ex_data_5_agg['month'] == m])

        ex_4_m = ex_data_4_agg[ex_data_4_agg['month'] == m][val_col].T.values
        ex_5_m = ex_data_5_agg[ex_data_5_agg['month'] == m][val_col].T.values
        era_diff_m = list(map(lambda x, y: mannwhitneyu(x, y, alternative='two-sided'), ex_4_m, ex_5_m))
        era_diff_m_df = pd.DataFrame(era_diff_m)
        era_diff_m_df['p < 0.050'] = era_diff_m_df['pvalue'].apply(lambda x: True if x < 0.025 else False)
        era_diff_m_df['r'] = era_diff_m_df['statistic'].apply(calc_r, args=[n1, n2])
        era_diff_m_df.insert(0, 'name', val_col)
        print(era_diff_m_df)


def check_stationariness(data):
    val_col = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    cp_data = data.copy()
    cp_data = cp_data[EX_COL]
    ex_data_4 = cp_data[(cp_data['era'] == 4) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(2022, 12, 31))]
    ex_data_5 = cp_data[(cp_data['era'] == 5) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(2023, 12, 31))]
    ex_data_4_agg = ex_data_4.groupby(['date']).agg(AGG_D).reset_index()
    ex_data_5_agg = ex_data_5.groupby(['date']).agg(AGG_D).reset_index()

    ex_data_4_agg['month'] = ex_data_4_agg['date'].apply(lambda x: x.month)
    ex_data_5_agg['month'] = ex_data_5_agg['date'].apply(lambda x: x.month)

    # 時系列データの変動成分の分解
    # トレンド成分、季節成分、残差成分
    # div_result = seasonal_decompose(ex_data_4_agg['amount_g'].values, model='multiplicative', period=4)
    # div_result = STL(ex_data_4_agg['amount_g'].values, period=4, robust=True).fit()
    # div_result.plot()
    # plt.show()


    # ADF検定：時系列データが単位根過程(非定常過程)か検定
    result = adfuller(ex_data_4_agg['amount_g'].values)
    print(result)

    # 有意水準5%
    alpha = 0.05

    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data = cp_data[(cp_data['era'] == e) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(w_c, 12, 31))]
        ex_data_agg = ex_data.groupby(['date']).agg(AGG_D).reset_index()
        ex_data_agg['month'] = ex_data_agg['date'].apply(lambda x: x.month)
        print(ex_data_agg)
        for cont in val_col:
            result = adfuller(ex_data_agg[cont].values)
            if result[1] < alpha:
                tp = 'steady'
            else:
                tp = 'unsteady'
            print(f'r: {e}, cont: {cont}, ADF Statistic: {result[0]}, p-value: {result[1]}, type: {tp}')

        # 月ごとに各数値データを
        # for m in range(4, 13):
        #     if m == 8:
        #         continue
        #     for cont in val_col:
        #         ex_data_agg_m = ex_data_agg[(ex_data_agg['month'] == m)]
        #         # print(ex_data_agg_m)
        #         result_m = adfuller(ex_data_agg_m[cont].values)
        #         if result_m[1] < alpha:
        #             tp = 'steady'
        #         else:
        #             tp = 'unsteady'
        #         print(f'r: {e}, month: {m}, cont: {cont}, ADF Statistic: {result_m[0]}, p-value: {result_m[1]}, type: {tp}')

        # 各数値データを月ごと
        # steady = 0
        # unsteady = 0
        # for cont in val_col:
        #     for m in range(4, 13):
        #         if m == 8:
        #             continue
        #         ex_data_agg_m = ex_data_agg[(ex_data_agg['month'] == m)]
        #         # print(ex_data_agg_m)
        #         result_m = adfuller(ex_data_agg_m[cont].values)
        #         if result_m[1] < alpha:
        #             tp = 'steady'
        #             steady += 1
        #         else:
        #             tp = 'unsteady'
        #             unsteady += 1
        #         plt.plot(list(range(0, len(ex_data_agg_m[cont]))), ex_data_agg_m[cont].values)
        #         print(f'r: {e}, month: {m}, cont: {cont}, ADF Statistic: {result_m[0]}, p-value: {result_m[1]}, type: {tp}')
        #         plt.show()
        # print(f'steady: {steady}, unsteady: {unsteady}')


def manipulate_unsteady_and_plot_correlogram(data):
    val_col = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    cp_data = data.copy()

    alpha = 0.05

    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data = cp_data[(cp_data['era'] == e) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(w_c, 10, 9))]  # 10, 9
        ex_data_agg = ex_data.groupby(['date']).agg(AGG_D).reset_index()
        ex_data_agg['month'] = ex_data_agg['date'].apply(lambda x: x.month)

        # print(ex_data_agg)
        # print(len(ex_data_agg[ex_data_agg['date'] <= datetime.date(w_c, 10, 15)]))
        # print(ex_data_agg[ex_data_agg['month'] == 10]['date'].apply(lambda x: x.strftime(format='%A')))

        # トレンド成分の削除
        diff_df = ex_data_agg.diff(1).dropna()
        # print(diff_df)

        # 季節成分の削除
        # (20日)で試す？p値が一番小さくなる？
        span = 20
        diff_df_m = ex_data_agg.diff(span).dropna()
        # print(diff_df_m)

        # トレンド成分->季節成分の削除
        span_t = span + 1
        diff_df_tm = diff_df.diff(span).dropna()

        print(f'-------------{span} days-------------')
        for cont in val_col:
            ex_d = ex_data_agg[cont]
            d_di = diff_df[cont]
            d_m = diff_df_m[cont]
            d_tm = diff_df_tm[cont]
            # 自然対数(loge, ln)変換
            ex_d_ln = ex_data_agg[cont].apply(np.log1p)

            result_ex = adfuller(ex_d.values)
            result_di = adfuller(d_di.values)
            result_m = adfuller(d_m.values)
            result_tm = adfuller(d_tm.values)
            result_ln = adfuller(ex_d_ln.values)

            if result_ex[1] < alpha:
                tp = 'steady'
            else:
                tp = 'unsteady'
            print(f'r: {e}, cont: {cont}, original, span: 0, ADF Statistic: {result_ex[0]}, p-valule: {result_ex[1]}, type: {tp}')

            if result_di[1] < alpha:
                tp = 'steady'
            else:
                tp = 'unsteady'
            print(f'r: {e}, cont: {cont}, trend only, span: 1, ADF Statistic: {result_di[0]}, p-valule: {result_di[1]}, type: {tp}')

            if result_m[1] < alpha:
                tp = 'steady'
            else:
                tp = 'unsteady'
            print(f'r: {e}, cont: {cont}, season only, span: {span}, ADF Statistic: {result_m[0]}, p-valule: {result_m[1]}, type: {tp}')

            if result_tm[1] < alpha:
                tp = 'steady'
            else:
                tp = 'unsteady'
            print(f'r: {e}, cont: {cont}, trend->season, span: {span_t}, ADF Statistic: {result_tm[0]}, p-valule: {result_tm[1]}, type: {tp}')

            if result_ln[1] < alpha:
                tp = 'steady'
            else:
                tp = 'unsteady'
            print(f'r: {e}, cont: {cont}, original->log, span: 0, ADF Statistic: {result_ln[0]}, p-valule: {result_ln[1]}, type: {tp}')
            # plt.plot(list(range(0, len(ex_d))), ex_d.values)
            # plt.show()

            # ex_d, ex_d_ln
            # acf1 = plot_acf(ex_d.values, lags=47)
            # pacf1 = plot_pacf(ex_d.values, lags=47)
            # acf2 = plot_acf(ex_d_ln.values, lags=40)
            # pacf2 = plot_pacf(ex_d_ln.values, lags=40)
            # d_di, d_m, d_tm
            # acf3 = plot_acf(d_di.values, lags=40)
            # pacf3 = plot_pacf(d_di.values, lags=40)
            # plt.show()

            # seasonal_decomposeでトレンド成分、季節成分、残差を分解
            # res = seasonal_decompose(ex_d, period=30)
            # res.plot()
            # plt.show()
            # STLでトレンド成分、季節成分、残差を分解
            # res_stl = STL(ex_d, period=50).fit()  # trend=91
            # res_stl.plot()

            # res_stl_o = res_stl.observed
            # res_stl_t = res_stl.trend
            # res_stl_s = res_stl.seasonal
            # res_stl_r = res_stl.resid
            # res_stl_o.plot()
            # res_stl_t.plot()
            # res_stl_s.plot()
            # res_stl_r.plot()
            # plt.title(f'{cont}')
            # plt.ylabel('val')
            # plt.xlabel('days')
            # plt.legend()
            # plt.show()

            # フーリエ変換によりスペクトル分析を行う
            # 高速フーリエ変換FFT(離散フーリエ変換DFTの高速版)
            y = ex_d
            n = len(y)
            t = 1.0
            # 高速フーリエ変換
            yf = np.fft.fft(y)
            xf = np.fft.fftfreq(n, d=t)[:n//2]

            # パワースペクトルを計算
            power_spectrum = 2 / n * np.abs(yf[0:n//2])

            # パワースペクトルのピーク
            peaks = argrelextrema(power_spectrum, np.greater)

            # ピークの周波数とパワーを取得
            peak_freqs = xf[peaks]
            peak_powers = power_spectrum[peaks]

            # ピークの周期を計算
            peak_periods = 1 / peak_freqs

            # プロット
            # plt.plot(xf, power_spectrum)
            # plt.plot(peak_freqs, peak_powers, 'ro')
            # plt.grid()
            # plt.title(f'power spectrum of {cont} with peaks')
            # plt.xlabel('freaquency [1/day]')
            # plt.ylabel('power')
            # plt.show()

            peak_df = pd.DataFrame({'frequency': peak_freqs,
                                    'power': peak_powers,
                                    'periods': peak_periods})
            peak_df = peak_df.sort_values('power', ascending=False)
            print(cont)
            print(peak_df)
            peak_df = peak_df.sort_values('power', ascending=True)
            plt.barh(list(range(0, len(peak_df))), peak_df['power'], tick_label=peak_df['periods'])
            plt.xlabel('power')
            plt.ylabel('periods')
            plt.title(f'power spectrum of {cont}')
            plt.tight_layout()
            plt.show()


def to_standarize(data):
    n = len(data)
    mean = np.mean(data)
    var = sum([(x - mean)**2 for x in data]) / (n - 1)
    std = var ** 0.5
    z = [(x - mean) / std for x in data]
    return z


def check_significance_fft_signal(data):
    val_col = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    cp_data = data.copy()

    alpha = 0.05
    n_bootstrap = 1000
    lower_percentile = 2.5
    upper_percentile = 97.5

    def optimal_block_size(y, n_bootstrap, lower_percentile, upper_percentile):
        # ブロックサイズを指定(何個のデータで1ブロックとするか)
        block_size_range = list(range(2, 146))

        # 各ブロックサイズでブートストラップを実施し、信頼区間の平均を記録
        average_interval_widths = []
        for block_size in block_size_range:
            bootstrap_powers = np.zeros((n_bootstrap, len(y)//2))
            for i in range(n_bootstrap):
                bootstrap_sample = []
                while len(bootstrap_sample) < len(y):
                    start_index = np.random.randint(0, len(y) - block_size + 1)
                    bootstrap_sample += list(y[start_index: start_index + block_size])
                bootstrap_sample = bootstrap_sample[:len(y)]

                bootstrap_yf = np.fft.fft(bootstrap_sample)
                bootstrap_powers[i, :] = 2 / len(y) * np.abs(bootstrap_yf[0:len(y)//2])

            lower_bound = np.percentile(bootstrap_powers, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_powers, upper_percentile, axis=0)

            average_interval_width = np.mean(upper_bound - lower_bound)
            average_interval_widths.append(average_interval_width)

        # 最適なブロックサイズを選択
        optimal_block_size = block_size_range[np.argmin(average_interval_widths)]

        return optimal_block_size

    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data = cp_data[(cp_data['era'] == e) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(w_c, 12, 31))]  # 10, 9
        ex_data_agg = ex_data.groupby(['date']).agg(AGG_D).reset_index()
        ex_data_agg['month'] = ex_data_agg['date'].apply(lambda x: x.month)
        ex_data_agg[val_col] = ex_data_agg[val_col].apply(to_standarize, axis=0)

        for cont in val_col:
            ex_d = ex_data_agg[cont]

            # フーリエ変換によりスペクトル分析を行う
            # 高速フーリエ変換FFT(離散フーリエ変換DFTの高速版)
            y = ex_d
            n = len(y)
            t = 1.0
            # 高速フーリエ変換
            yf = np.fft.fft(y)
            xf = np.fft.fftfreq(n, d=t)[:n//2]

            # block_size = optimal_block_size(y, n_bootstrap, lower_percentile, upper_percentile)
            # print(block_size)


            # パワースペクトルを計算
            power_spectrum = 2 / n * np.abs(yf[0:n//2])

            # パワースペクトルのピーク
            peaks = argrelextrema(power_spectrum, np.greater)

            # ピークの周波数とパワーを取得
            peak_freqs = xf[peaks]
            peak_powers = power_spectrum[peaks]

            # ピークの周期を計算
            peak_periods = 1 / peak_freqs

            block_size = 145
            bootstrap_powers = np.zeros((n_bootstrap, n//2))
            for i in range(n_bootstrap):
                bootstrap_sample = []
                while len(bootstrap_sample) < n:
                    start_index = np.random.randint(0, n - block_size + 1)
                    bootstrap_sample += list(y[start_index: start_index + block_size])
                bootstrap_sample = bootstrap_sample[:n]

                bootstrap_yf = np.fft.fft(bootstrap_sample)
                bootstrap_powers[i, :] = 2 / n * np.abs(bootstrap_yf[0:n//2])
            # 信頼区間を計算
            lower_bound = np.percentile(bootstrap_powers, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_powers, upper_percentile, axis=0)


            # ピークが有意かどうかを判断
            significant = np.logical_or(peak_powers < lower_bound[peaks], peak_powers >
                                          upper_bound[peaks])

            # プロット
            # plt.plot(xf, power_spectrum)
            # plt.plot(peak_freqs, peak_powers, 'ro')
            # plt.grid()
            # plt.title(f'power spectrum of {cont} with peaks')
            # plt.xlabel('freaquency [1/day]')
            # plt.ylabel('power')
            # plt.show()

            peak_df = pd.DataFrame({'frequency': peak_freqs,
                                    'power': peak_powers,
                                    'periods': peak_periods,
                                    'significant': significant})
            peak_df = peak_df.sort_values('power', ascending=False)
            print(cont)
            print(peak_df)
            peak_df = peak_df.sort_values('power', ascending=True)
            # plt.barh(list(range(0, len(peak_df))), peak_df['power'], tick_label=peak_df['periods'])
            # plt.xlabel('power')
            # plt.ylabel('periods')
            # plt.title(f'power spectrum of {cont}')
            # plt.tight_layout()
            # plt.show()


class Spectra(object):

    def __init__(self, t, f, time_unit):
        """スペクトル分析を行う

        Args:
            t (_type_): 時間軸の値
            f (_type_): データの値
            time_unit (_type_): 時間軸の単位
        """
        assert t.size == f.size
        assert np.unique(np.diff(t)).size == 1
        self.t = t
        self.f = f
        self.time_unit = time_unit
        T = (t[1] - t[0]) * t.size
        self.period = 1.0 / (np.arange(t.size / 2)[1:] / T)

        # パワースペクトル密度を計算
        f = f - np.average(f)
        F = fftpack.fft(f)
        self.po = np.abs(F[1:(t.size // 2)]) ** 2 / T

    # def draw_with_time(self, fsizex=8, fsizey=6, print_flg=True, threshold=3.0):
    #     fig, ax = plt.subplots(figsize=(fsizex, fsizey))
    #     ax.set_yscale('log')
    #     ax.set_xscale('log')
    #     ax.set_xlabel(self.time_unit)
    #     ax.set_ylabel('power spectrum density')
    #     ax.plot(self.period, self.po)
    #     if print_flg:
    #         dominant_periods = self.period[self.po > threshold]
    #         print(dominant_periods, self.time_unit + ' components are dominant!')
    #         for dominant_period in dominant_periods:
    #             plt.axvline(x=dominant_period, linewidth=0.5, color='k')
    #             ax.text(dominant_period, threshold, str(round(dominant_period, 3)))

    #     return plt

    def draw_with_time(self, title_words, fsizex=8, fsizey=6, print_flg=True, threshold=1.0):
        titles = [f'r:{title_words}, group:{GRP}, {text}' for text in ['menu_counts', 'spectra']]
        fig = make_subplots(rows=2, cols=1, subplot_titles=titles)
        trace_count = go.Scatter(x=self.t, y=self.f, mode='lines', name=f'r{title_words}_{GRP}_menu_contents_counts')
        trace_spectra = go.Scatter(x=self.period, y=self.po, mode='lines', name=f'r{title_words}_{GRP}_spectra')
        fig.add_trace(trace_count, row=1, col=1)
        fig.add_trace(trace_spectra, row=2, col=1)
        if print_flg:
            dominant_periods = self.period[self.po > threshold]
            print(dominant_periods, self.time_unit + ' component are dominant')
            for dominant_period in dominant_periods:
                trace_dominant = go.Scatter(x=[dominant_period]*10, y=list(range(0, 10)), mode='lines', line=dict(color='black', width=0.5))
                fig.add_trace(trace_dominant, row=2, col=1)
        fig.update_layout(**{'yaxis1': dict(range=[1, 5])},
                          **{'xaxis2': dict(type='log', exponentformat='e')},
                          **{'yaxis2': dict(type='log', exponentformat='e')},
                          )
        return fig


def manual_fft(data):
    val_col = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
    cp_data = data.copy()
    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data = cp_data[(cp_data['era'] == e) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(w_c, 12, 31))]  # 10, 9
        ex_data_agg = ex_data.groupby(['date']).agg(AGG_D).reset_index()
        ex_data_agg['month'] = ex_data_agg['date'].apply(lambda x: x.month)
        ex_data_agg[val_col] = ex_data_agg[val_col].apply(to_standarize, axis=0)

        for cont in val_col:
            ex_d = ex_data_agg[cont]
            n = len(ex_d)
            t = np.arange(0, n)
            f = ex_d.values

            # plt.figure(figsize=(20, 6))
            # plt.plot(t, f)
            # plt.xlim(0, n)
            # plt.xlabel('day')
            # plt.show()

            # 周期の描画
            print(f'r:{e}, grp:{GRP}, cont:{cont}')
            spectra = Spectra(t, f, 'day')
            plt_s = spectra.draw_with_time(title_words=e)
            plt_s.show()


# 献立に和洋中のフラグを付けたい
# 文字列を分析する
def menu_kinds_analysis(data):
    cols = ['era', 'area_group', 'date', 'menu']
    cp_data = data.copy()

    def count_drink(text_set: Tuple[str]):
        text_l = list(text_set)
        count = 0
        for tx in text_l:
            if '乳' in tx:
                count += 1
        return count

    def extract_menu(text_set: Tuple[str], word):
        text_l = list(text_set)
        ex_text_l = [tx for tx in text_l if word in tx]
        ex_text = ''
        if len(ex_text_l) == 1:
            ex_text = ex_text_l[0]
        elif len(ex_text_l) > 1:
            raise Exception(f'同じ単語が複数の献立に含まれます: {word} -> {ex_text_l}')
        return ex_text

    def without_milk(text_set: Tuple[str]):
        text_l = list(text_set)
        ex_text_l = [tx for tx in text_l if '乳' not in tx]
        return ex_text_l


    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data = cp_data[(cp_data['era'] == e) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(w_c, 12, 31))]  # 10, 9
        ex_ = ex_data.groupby(['date']).agg({'area_group': 'count', 'menu': set}).reset_index()
        # ex_['drink'] = ex_['menu'].apply(count_drink)
        ex_['menu_without_milk'] = ex_['menu'].apply(without_milk)
        ex_['menu_counts_without_milk'] = ex_['menu_without_milk'].apply(len)
        # drink_sum = ex_['drink'].value_counts()
        data_sum = len(ex_)
        print(f'r:{e}, area_group:{GRP}')
        # print(ex_)
        # print(f'data_sum:{data_sum}')
        # print(drink_sum)
        print(ex_['menu_counts_without_milk'].value_counts())

        def check_rice(data_l):
            result = True
            for m in data_l:
                if 'パン' in m or 'ごはん' in m:
                    result = False
            return result
        rice_bool = ex_[(ex_['menu_counts_without_milk'] == 3)]['menu_without_milk'].apply(check_rice)
        rice_menu = ex_[(ex_['menu_counts_without_milk'] == 3)][rice_bool]
        # print(f'rice_sum: {len(rice_menu)}')
        # print(rice_menu)

        # スペクトル分析
        n = len(ex_)
        t = np.arange(0, n)
        f = ex_['menu_counts_without_milk'].values
        # print(f'r:{e}, grp:{GRP}')
        spectra = Spectra(t, f, 'days')
        plt_fft = spectra.draw_with_time(title_words=e, threshold=1.6)
        plt_fft.show()


def calc_approximate_line(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    cov = sum([(xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)]) / (n - 1)
    x_var = sum([(xi - x_mean) ** 2 for xi in x]) / (n - 1)
    slope = cov / x_var
    intercept = y_mean - slope * x_mean
    print(f'slope:{slope}, intercept:{intercept}, y_mean:{y_mean}')

    appro = [slope * xi + intercept for xi in x]
    return appro


def without_milk(text_set: Tuple[str]):
    text_l = list(text_set)
    ex_text_l = [tx for tx in text_l if '乳' not in tx]
    return ex_text_l


def menu_value_analysis(data):
    cp_data = data.copy()
    agg_d = AGG_D.copy()
    agg_d['menu'] = set

    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data = cp_data[(cp_data['era'] == e) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(w_c, 12, 31))]
        ex_data_agg = ex_data.groupby(['date']).agg(agg_d).reset_index()
        ex_data_agg['without_milk'] = ex_data_agg['menu'].apply(without_milk)
        ex_data_agg['menu_counts'] = ex_data_agg['without_milk'].apply(len)

        content = 'sodium_mg'
        menu_counts = 4
        print(f'r: {e}, content: {content}, menu_count: {menu_counts}')
        ex_ = ex_data_agg[(ex_data_agg['menu_counts'] == menu_counts)]
        appro = calc_approximate_line(list(range(0, len(ex_))), ex_[content])

        fig = go.Figure()
        trace = go.Scatter3d(x=list(range(0, len(ex_))), y=ex_['menu_counts'].values, z=ex_[content].values, mode='markers')
        trace_appro = go.Scatter3d(x=list(range(0, len(ex_))), y=ex_['menu_counts'].values, z=appro, mode='lines')
        fig.add_trace(trace)
        fig.add_trace(trace_appro)
        fig.update_layout(scene=dict(
                                    xaxis=dict(title='days'),
                                    yaxis=dict(title='menu_counts'),
                                    zaxis=dict(title=f'{content}')
                                    ))
        fig.show()

        print(len(ex_data_agg))
        print(ex_data_agg)


def extract_specified_contents(menu_list, specified_word_list):
    menu_type = 0

    # search_l = [r'ごはん', r'パン|フランス']

    # def search_specific_word(search_w, word_list):
    #     map_l = map(lambda m: True if re.search(search_w, m) else False, word_list)
    #     return sum(map_l)
    # bool_l = [search_specific_word(s, menu_list) for s in search_l]
    # print(bool_l)

    for type_num, specified_word in enumerate(specified_word_list):
        ex_list = [m for m in menu_list if re.search(specified_word, m)]
        if len(ex_list) > 0:
            menu_type = type_num
            break
        else:
            menu_type = len(specified_word_list)
    return menu_type


def menu_contents_analysis(data):
    cp_data = data.copy()
    agg_d = AGG_D.copy()
    agg_d['menu'] = set

    menu_counts = 3
    specified_word_list = [r'こぎつね|おこわ|たきこみ', r'カレー', r'丼|ライス|そぼろご', r'ごはん', r'パン|フランス', r'そば|うどん|スパゲ|めん|メン']

    for e in [4, 5]:
        if e == 4:
            w_c = 2022
        else:
            w_c = 2023
        ex_data = cp_data[(cp_data['era'] == e) & (cp_data['area_group'] == GRP) & (cp_data['date'] <= datetime.date(w_c, 12, 31))]
        ex_data_agg = ex_data.groupby(['date']).agg(agg_d).reset_index()
        ex_data_agg['without_milk'] = ex_data_agg['menu'].apply(without_milk)
        ex_data_agg['menu_counts'] = ex_data_agg['without_milk'].apply(len)
        ex_data_agg['menu_type'] = ex_data_agg['without_milk'].apply(extract_specified_contents, args=[specified_word_list])
        print(ex_data_agg['menu_type'].value_counts())
        # print(ex_data_agg[(ex_data_agg['menu_counts'] == menu_counts) & (ex_data_agg['menu_type'] == 0)].drop(columns='menu'))
        print(ex_data_agg[(ex_data_agg['menu_type'] == 6)].drop(columns='menu'))

