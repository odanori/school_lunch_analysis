import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    output_num_graph(data)
