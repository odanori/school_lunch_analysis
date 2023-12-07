import datetime
import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


class Preprocessor(ABC):
    """前処理に関する基底クラス
       前処理内容をそれぞれ関数として追加する
       processに各前処理の関数をまとめる
    """

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class ValuesDeleter(Preprocessor):
    """不要な値を削除するクラス
    """

    def __init__(self) -> None:
        # csvデータ中の不明データのレコードを出力する
        self.unknown_log_path = Path('../unknown_log')

    def delete_nan_row(self, data: pd.DataFrame) -> pd.DataFrame:
        """Nanで入力された行を削除する

        Args:
            data (pd.DataFrame): csvデータ
                         例:A日付  献立名   材料名   分量（g）  エネルギー（kcal）  たんぱく質（g）  脂質（g）  ナトリウム（mg）
                        2023/11/1   黒パン  黒パン    50       236                7.1            3.8         358
                        2023/11/1   牛乳    牛乳     206       126                6.8            7.8         84
                        nan         nan     nan      nan      nan                nan            nan         nan

        Returns:
            pd.DataFrame: データ行すべてがnanのレコードを削除したデータ
                          例:A日付  献立名   材料名   分量（g）  エネルギー（kcal）  たんぱく質（g）  脂質（g）  ナトリウム（mg）
                        2023/11/1   黒パン  黒パン    50       236                7.1            3.8         358
                        2023/11/1   牛乳    牛乳     206       126                6.8            7.8         84
        """
        cp_data = data.copy()
        # データ読み込み時にera, area_group, monthが入力されるのでthresh=4
        cp_data.dropna(thresh=4, axis=0, inplace=True)
        del data
        return cp_data

    def delete_unname_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """データがないはずのカラム(Unnamedカラム)が存在する場合は削除する
           カラム名がないにもかかわらず、何らかのデータがあった場合は
           out_of_range_logファイルに記載
           例: out_of_range_logファイルの中身
               era group     日付       献立名     材料名 分量（g）... ナトリウム（mg）Unnamed: 8 Unnamed: 9
                 4   c  2022/9/6  あじのこはく揚げ   油    35      ...     0.0          NaN     2022/6/16
                  →データがないはずのUnnamed:9カラムに日付らしきものがある

        Args:
            data (pd.DataFrame): csvデータ
                      例 era group     日付       献立名     材料名 分量（g）...  ナトリウム（mg）  Unnamed: 8 Unnamed: 9
                           4   c  2022/11/1      麦ごはん    米     70      ...        1.0          NaN        nan

        Returns:
            pd.DataFrame: 引数のデータからUnnamedカラムが削除処理されたもの
                    例 era group     日付       献立名     材料名 分量（g）...  ナトリウム（mg）
                           4   c  2022/11/1      麦ごはん    米     70    ...      1.0
        """
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
        """不要データの削除をまとめて行う関数

        Args:
            data (pd.DataFrame): csvデータ

        Returns:
            pd.DataFrame: 不要データが削除されたデータ
        """
        processed_data = self.delete_nan_row(data)
        processed_data = self.delete_unname_columns(processed_data)
        return processed_data


class ValuesRenamer(Preprocessor):

    def rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """カラム名を日本語から英語に変更する

        Args:
            data (pd.DataFrame): csvデータ
                        例:A日付  献立名   材料名   分量（g） ...

        Returns:
            pd.DataFrame: カラム名を変更したデータ
                        例:date  menu  ingredient  amount_g ...
        """
        # TODO: カラム名を指定して変更できるようにしたほうが良いか
        cp_data = data.copy()
        half_width_cols = {}
        english_cols = {
            '献立名': 'menu', '材料名': 'ingredient', '分量(g)': 'amount_g', 'エネルギー(kcal)': 'carolies_kcal',
            'たんぱく質(g)': 'protein_g', '脂質(g)': 'fat_g', 'ナトリウム(mg)': 'sodium_mg'
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
        """データ内の値名称の変更を行う

        Args:
            data (pd.DataFrame): csvデータ

        Returns:
            pd.DataFrame: 各種名称変更されたデータ
        """
        processed_data = self.rename_columns(data)
        return processed_data


class ValuesManipulator(Preprocessor):

    def manipulate_nan_in_numerical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Nanが入力された数字データ入力範囲は0置換を行う

        Args:
            data (pd.DataFrame): csvデータ
                      例:  A日付       献立名        材料名  分量 （g）
                          2023/10/29  炊き込みご飯     水    Nan

        Returns:
            pd.DataFrame: 文字列での数字が数字型に変換されたデータ
                      例:  A日付       献立名        材料名  分量 （g）
                          2023/10/29  炊き込みご飯     水    0
        """
        cp_data = data.copy()
        # TODO: 材料ごとの欠損値対応は後程実装(現状は 材料名=水 以外欠損値なし)

        cp_data.iloc[:, 6:] = cp_data.iloc[:, 6:].fillna(0)
        del data
        return cp_data

    def manipulate_str_numerical_value(self, data: pd.DataFrame) -> pd.DataFrame:
        """文字列で入力された数字を数字型floatに変換する
           '120~130'のような範囲で入力された値はcalc_mean_value関数で平均値に置換する

        Args:
            data (pd.DataFrame): csvデータ
                      例:  A日付       献立名        材料名  分量 （g）
                          2023/10/31  ごはん           米    '100'
                          2023/10/31  ワンタンスープ    水    120~130

        Returns:
            pd.DataFrame: 文字列での数字が数字型に変換されたデータ
                      例:  A日付       献立名        材料名  分量 （g）
                          2023/10/31  ごはん           米    100.0
                          2023/10/31  ワンタンスープ    水    125.0
        """
        cp_data = data.copy()

        def calc_mean_value(str_range_value: str) -> str:
            split_value = re.split(r"~|～", str_range_value)
            mean_value = np.mean(list(map(float, split_value)))
            return str(mean_value)
        cp_data = cp_data.astype(str)
        cp_data.iloc[:, 6:] = cp_data.iloc[:, 6:].apply(lambda x: list(map(calc_mean_value, x)), axis=0)
        cp_data.iloc[:, 6:] = cp_data.iloc[:, 6:].astype(float)
        del data
        return cp_data

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ入力範囲内の非数(Nan)をまとめて処理する

        Args:
            data (pd.DataFrame): csvデータ

        Returns:
            pd.DataFrame: 非数が処理されたデータ
        """
        preprocessed_data = self.manipulate_nan_in_numerical_columns(data)
        preprocessed_data = self.manipulate_str_numerical_value(preprocessed_data)

        return preprocessed_data


class DatetimeChanger(Preprocessor):

    def change_to_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        """日付をdatetime型に変換する

        Args:
            data (pd.DataFrame): csvデータ

        Returns:
            pd.DataFrame: 日付をdatetime型に変換したデータ
        """
        cp_data = data.copy()
        datetime_data = pd.to_datetime(cp_data.iloc[:, 3])
        cp_data.iloc[:, 3] = datetime_data
        del data
        return cp_data

    def add_era_name(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ内の日付から元号を照合し、対応した元号名を入力した新たなカラムを追加

        Args:
            data (pd.DataFrame): 日付のカラムがdatetime型に変換されたデータ
                        例: ... date      menu ... sodium_mg
                               2023/10/31 ごはん     0

        Returns:
            pd.DataFrame: 元号カラム(era_name)が追加されたデータ
                        例: ... date       menu ... sodium_mg  era_name
                               2023/10/31  ごはん     0         令和
        """
        def check_era(date: datetime.datetime):
            era_dict = {'令和': datetime.datetime(2019, 5, 1),
                        '平成': datetime.datetime(1989, 1, 8),
                        '昭和': datetime.datetime(1926, 12, 25)}
            if era_dict['令和'] <= date:
                era_name = '令和'
            elif era_dict['平成'] <= date:
                era_name = '平成'
            elif era_dict['昭和'] <= date:
                era_name = '昭和'
            else:
                era_name = None
            return era_name
        cp_data = data.copy()
        cp_data['era_name'] = cp_data.iloc[:, 3].apply(lambda x: check_era(x))
        del data
        return cp_data

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """時間に関する値の処理を行う

        Args:
            data (pd.DataFrame): csvデータ

        Returns:
            pd.DataFrame: 時間の値が処理されたデータ
        """
        preprocessed_data = self.change_to_datetime(data)
        preprocessed_data = self.add_era_name(preprocessed_data)
        return preprocessed_data


class DataProcessor:
    """各前処理クラスをまとめて、それぞれ実行するクラス
    """
    def __init__(self, processors: List[Preprocessor]) -> None:
        self.processors = processors

    def process_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        processed_data = input_data.copy()
        for processor in self.processors:
            processed_data = processor.process(processed_data)
        del input_data
        return processed_data


def data_processor(data: pd.DataFrame) -> pd.DataFrame:
    """各前処理クラスを順番に実行する関数
       前処理クラスを追加した場合はprocessorsに追加する

    Args:
        data (pd.DataFrame): csvデータ

    Returns:
        pd.DataFrame: 各種前処理が完了したデータ
    """
    processors = [ValuesDeleter(), ValuesRenamer(), ValuesManipulator(), DatetimeChanger()]
    result = DataProcessor(processors).process_data(data)
    return result
