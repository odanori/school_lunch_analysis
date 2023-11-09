from pathlib import Path

import pandas as pd

from analyzer.config import Config


class Preprocessing():
    """読み込みデータ一つに対して前処理を行うクラス
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.out_of_range_log_path = Path('./out_of_range')

    def delete_nan_row(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ行のすべてがnanのレコードを削除する
           例:日付~栄養を示すデータがすべてnanの場合、削除
            A日付  献立名 材料名 分量（g）  エネルギー（kcal）  たんぱく質（g）  脂質（g）  ナトリウム（mg）
            nan    nan    nan    nan        nan                 nan            nan         nan
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
        data.dropna(how='all', axis=0, inplace=True)
        return data

    def add_era_and_group(self, data: pd.DataFrame, era: int, group: int) -> pd.DataFrame:
        """データに元号と地域グループ番号を追加

        Args:
            data (pd.DataFrame): delete_nan_rowで処理されたデータ
                           例:A日付  献立名   材料名   分量（g）  ...
                         2023/11/1   黒パン  黒パン    50
            era (int): 元号
            group (int): 地域グループ番号

        Returns:
            pd.DataFrame: 元号、地域グループ番号が追加されたデータ
                        例:era group    A日付  献立名   材料名   分量（g）  ...
                            5    a  2023/11/1   黒パン  黒パン    50
        """
        data.insert(0, 'era', era)
        data.insert(1, 'group', group)
        return data

    def separate_column_name(self, data: pd.DataFrame) -> pd.DataFrame:
        """データごとに同じ意味で異なるカラム名を統一する
           カラム名で共通する部分をjsonファイルで指定し、その部分だけで統一

        Args:
            data (pd.DataFrame): csvデータ

        Raises:
            Exception: 変更対象のカラムが複数ある場合は例外処理
                       #TODO: 複数のカラム名変更に対応できるようにする→カラム名を英語対応する

        Returns:
            pd.DataFrame: 引数のデータからカラム名が変更処理されたもの
                          例: 地域グループ番号Aのデータのカラム
                              A日付, 献立, 材料名,...
                              地域グループ番号Bのデータのカラム
                              B日付, 献立, 材料名,...
                              変更後
                              日付, 献立, 材料名,...
        """
        # FIXME: 現状、日付のみの対応
        columns = data.columns.values
        col_idx_l = [i for i, col in enumerate(
            columns) if col.endswith(self.config.rename_col)]
        if len(col_idx_l) > 1:
            raise Exception(
                f'変更対象:{self.config.rename_col}が1カラムより多く存在します。確認ください:\n{data}')
        rename_idx = col_idx_l[0]
        rename_col = {columns[rename_idx]: self.config.rename_col}
        data.rename(columns=rename_col, inplace=True)
        return data

    def delete_unname_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """データがないはずのカラム(Unnamedカラム)が存在する場合は削除する
           カラム名がないにもかかわらず、何らかのデータがあった場合は
           out_of_range_logファイルに記載
           例: out_of_range_logファイルの中身
               era group     日付       献立名     材料名 分量（g）  エネルギー（kcal）  たんぱく質（g）  脂質（g）  ナトリウム（mg）  Unnamed: 8 Unnamed: 9
                 4   c  2022/9/6  あじのこはく揚げ   油    35         21.0              0.0            2.4            0.0          NaN       2022/6/16
                  →データがないはずのUnnamed:9カラムに日付らしきものがある

        Args:
            data (pd.DataFrame): csvデータ
                      例 era group     日付       献立名     材料名 分量（g）  エネルギー（kcal）  たんぱく質（g）  脂質（g）  ナトリウム（mg）  Unnamed: 8 Unnamed: 9
                           4   c  2022/11/1      麦ごはん    米     70         239               4.3             0.6            1.0          NaN        nan

        Returns:
            pd.DataFrame: 引数のデータからUnnamedカラムが削除処理されたもの
                    例 era group     日付       献立名     材料名 分量（g）  エネルギー（kcal）  たんぱく質（g）  脂質（g）  ナトリウム（mg）
                           4   c  2022/11/1      麦ごはん    米     70         239               4.3             0.6            1.0
        """
        columns = data.columns.values
        del_col_l = [col for col in columns if col.startswith('Unnamed')]
        if len(del_col_l) > 0:
            null_judge = data.loc[:, del_col_l].isnull().all()
            if False in null_judge.values:
                not_nan_record = data.loc[:, del_col_l].dropna(
                    how='all', axis=0)
                with open(self.out_of_range_log_path, 'a', encoding='utf-8_sig') as out_of_range_log:
                    print(data.loc[not_nan_record.index, :].to_string(index=False),
                          file=out_of_range_log)
            data.drop(columns=del_col_l, inplace=True)
        return data

    def data_formatting(self, data: pd.DataFrame, era: int, group: int) -> pd.DataFrame:
        """入力データ範囲外のエラーデータを削除する
           元号や地域グループ番号を追加する

        Args:
            data (pd.DataFrame): csvデータ
            era (int): 元号
            group (int): 地域グループ番号

        Returns:
            pd.DataFrame: 整形されたデータ
        """
        formatting_data = data.copy()
        formatting_data = self.delete_nan_row(formatting_data)
        formatting_data = self.add_era_and_group(formatting_data, era, group)
        formatting_data = self.separate_column_name(formatting_data)
        formatting_data = self.delete_unname_columns(formatting_data)
        del data
        return formatting_data

    def manipulate_nan_data(self, formatting_data: pd.DataFrame):
        # print(formatting_data.isnull())
        elements = formatting_data[formatting_data.isnull().any(
            axis=1)][self.config.nan_col].values
        elements = list(set(elements))
        # FIXME: 材料名が「水」以外の欠損値が出た場合は例外処理。材料に合わせた処理(入力ミスなのか、など)を検討し実装
        # 「水」は汁もの以外のレシピ内容で、料理ごとに測れるものでない、かつ栄養も0なので0埋め処理
        if elements[0] != '水':
            raise Exception('水以外の欠損があった場合の処理は後程実装')
        elif len(elements) > 1:
            raise Exception('水以外の欠損があった場合の処理は後程実装')
        else:
            pass
        # print(formatting_data[formatting_data.isnull().any(axis=1)])

    def data_preprocess(self, data: pd.DataFrame, era: int, group: int) -> pd.DataFrame:
        formatting_data = self.data_formatting(data, era, group)
        self.manipulate_nan_data(formatting_data)
        return formatting_data
