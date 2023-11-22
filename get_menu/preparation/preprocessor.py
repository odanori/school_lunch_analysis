import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class Preprocessor(ABC):

    @abstractmethod
    def process(self, data: pd.DataFrame):
        pass


class ValuesDeleter(Preprocessor):

    def __init__(self) -> None:
        # csvデータ中の不明データのレコードを出力する
        self.unknown_log_path = Path('../unknown_log')

    def delete_nan_row(self, data: pd.DataFrame) -> pd.DataFrame:
        cp_data = data.copy()
        cp_data.dropna(how='all', axis=0, inplace=True)
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

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        processed_data = self.delete_nan_row(data)
        processed_data = self.delete_unname_columns(data)
        return processed_data


class ValuesRenamer(Preprocessor):

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

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        processed_data = self.rename_columns(data)
        return processed_data


class ValuesComplementer(Preprocessor):

    def complement_nan_in_numerical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        cp_data = data.copy()
        # TODO: 材料ごとの欠損値対応は後程実装(現状は 材料名=水 以外欠損値なし)

        # missing_val_in_ingredient = data[data.isnull().any(axis=1)]['ingredient'].values
        # if len(missing_val_in_ingredient) > 1:
        #     raise Exception('欠損値のある材料が複数種類あります')
        # if missing_val_in_ingredient[0] != '水':
        #     raise Exception('水以外の材料に関する欠損値は未対応')

        cp_data.iloc[:, 6:].fillna(0, inplace=True)
        del data
        return cp_data

    def complement_str_numerical_value(self, data: pd.DataFrame) -> pd.DataFrame:
        cp_data = data.copy()

        def calc_mean_value(str_range_value: str) -> str:
            split_value = re.split(r"~|～", str_range_value)
            mean_value = np.mean(list(map(float, split_value)))
            return str(mean_value)

        cp_data.iloc[:, 6:] = cp_data.iloc[:, 6:].apply(lambda x: list(map(calc_mean_value, x)))
        del data
        return cp_data

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        preprocessed_data = self.complement_nan_in_numerical_columns(data)
        preprocessed_data = self.complement_str_numerical_value(preprocessed_data)
        return preprocessed_data


class DatetimeChanger(Preprocessor):

    def change_to_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        cp_data = data.copy()
        datetime_data = pd.to_datetime(cp_data.iloc[:, 2])
        cp_data.iloc[:, 2] = datetime_data
        del data
        return cp_data

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        preprocessed_data = self.change_to_datetime(data)
        return preprocessed_data
