import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


class Preprocessing():

    def __init__(self) -> None:

        # csvデータ中の不明データのレコードを出力する
        self.unknown_log_path = Path('../unknown_log')

    def delete_nan_row(self, data: pd.DataFrame) -> pd.DataFrame:
        cp_data = data.copy()
        cp_data.dropna(how='all', axis=0, inplace=True)
        del data
        return cp_data

    def add_era_and_group(self, data: pd.DataFrame, era: int, group: str) -> pd.DataFrame:
        cp_data = data.copy()
        cp_data.insert(0, 'era', era)
        cp_data.insert(1, 'group', group)
        del data
        return cp_data

    def rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:

        # TODO: カラム名を指定して変更できるようにしたほうが良いか
        cp_data = data.copy()
        half_width_cols = {}
        english_cols = {
            '献立名': 'menu', '材料名': 'ingredient', '分量(g)': 'amount[g]', 'エネルギー(kcal)': 'carolies[kcal]',
            'たんぱく質(g)': 'protein[g]', '脂質(g)': 'fat[g]', 'ナトリウム(mg)': 'sodium[mg]'
            }

        columns = cp_data.columns.values
        norm_columns = list(map(lambda x: unicodedata.normalize('NFKC', x), columns))
        date_idx = [i for i, col in enumerate(columns) if col.endswith('日付')]
        for col, norm_col in zip(columns, norm_columns):
            half_width_cols[col] = norm_col

        # 現状は日付のみ地域グループ番号が付与されており、地域グループ番号を除した「日付」で統一
        if len(date_idx) > 1:
            raise Exception('日付 が含まれるカラムが複数あります')

        half_width_cols[columns[date_idx[0]]] = 'date'
        cp_data.rename(columns=half_width_cols, inplace=True)
        cp_data.rename(columns=english_cols, inplace=True)

        del data
        return cp_data

    def delete_unname_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        cp_data = data.copy()
        columns = cp_data.columns.values
        unname_cols = [col for col in columns if col.startswith('Unnamed')]
        if len(unname_cols) > 0:
            judge_null = cp_data.loc[:, unname_cols].isnull().all()
            if False in judge_null.values:
                unknown_record = cp_data.loc[:, unname_cols].dropna(how='all', axis=0)
                with open(self.unknown_log_path, 'a', encoding='utf-8_sig') as unknown_log:
                    print(data.loc[unknown_record.index, :].to_string(index=False), file=unknown_log)
            cp_data.drop(columns=unname_cols, inplace=True)
        del data
        return cp_data

    def data_formatting(self, data: pd.DataFrame, era: int, group: str) -> pd.DataFrame:
        formatting_data = self.delete_nan_row(data)
        formatting_data = self.add_era_and_group(formatting_data, era, group)
        formatting_data = self.rename_columns(formatting_data)
        formatting_data = self.delete_unname_columns(formatting_data)
        return formatting_data

    def manipulate_nan_data(self, formatting_data: pd.DataFrame) -> pd.DataFrame:
        cp_data = formatting_data.copy()
        # TODO: 水 以外の材料名の各種値に欠損値が出た場合は例外処理。材料に合わせた処理(入力ミスなのか、など)を検討し実装
        missing_val_in_ingredient = formatting_data[formatting_data.isnull().any(axis=1)]['ingredient'].values
        if len(missing_val_in_ingredient) > 1:
            raise Exception('欠損値のある材料が複数種類あります')
        if missing_val_in_ingredient[0] != '水':
            raise Exception('水以外の材料に関する欠損値は未対応')

        cp_data.fillna(0, inplace=True)
        del formatting_data
        return cp_data

    def manipulate_str_data(self, formatting_data: pd.DataFrame) -> pd.DataFrame:

        def calc_mean_value(str_range_value: str) -> str:
            split_val = re.split(r"~|～", str_range_value)
            mean_val = np.mean(list(map(str, split_val)))

    def data_preprocessing(self, data: pd.DataFrame, era: int, group: str) -> pd.DataFrame:
        formatting_data = self.data_formatting(data, era, group)
        print(formatting_data)
        return formatting_data
