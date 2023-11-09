from pathlib import Path

import pandas as pd

from analyzer.config import Config


class Preprocessing():
    """読み込みデータ一つに対して前処理を行うクラス
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.out_of_range_log_path = Path('./out_of_range')

    def separate_column_name(self, data: pd.DataFrame) -> pd.DataFrame:
        """データごとに同じ意味で異なるカラム名を統一する
           カラム名で共通する部分をjsonファイルで指定し、その部分だけで統一

        Args:
            data (pd.DataFrame): csvから読み出したデータ

        Raises:
            Exception: 変更対象のカラムが複数ある場合は例外処理
                       #TODO: 複数のカラム名変更に対応できるようにする→カラム名を英語対応する

        Returns:
            pd.DataFrame: 引数のデータからカラム名が変更処理されたもの
                          例: 地域グループ番号Aのデータのカラム
                              A日付, 献立, 材料名,...
                              地域グループ番号Bのデータのカラム
                              B日付, 献立, 材料名,...
                              →日付カラムの先頭には地域グループ番号がつくが、どのデータの日付も同じ意味のため、
                              グループ番号と日付を分離し、日付カラムの名称を統一する
        """
        # FIXME: 現状、日付のみの対応
        columns = data.columns.values
        col_idx_l = [i for i, col in enumerate(
            columns) if col.endswith(self.config.rename_col)]
        if len(col_idx_l) > 1:
            raise Exception(
                f'変更対象:{self.config.rename_col}が1カラムより多く存在します。確認ください:\n{data}')
        rename_idx = col_idx_l[0]
        group = columns[rename_idx].replace(self.config.rename_col, '')
        rename_col = {columns[rename_idx]: self.config.rename_col}
        data.rename(columns=rename_col, inplace=True)
        data.insert(0, 'group', group.lower())
        return data

    def delete_unname_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """データがないはずのカラム(Unnamedカラム)が存在する場合は削除する
           カラム名がないにもかかわらず、何らかのデータがあった場合は
           out_of_range_logファイルに記載
           例: out_of_range_logファイルの中身
               group     日付       献立名     材料名 分量（g）  エネルギー（kcal）  たんぱく質（g）  脂質（g）  ナトリウム（mg）  Unnamed: 8 Unnamed: 9
                  c  2022/9/6  あじのこはく揚げ   油    35         21.0              0.0            2.4            0.0          NaN       2022/6/16
                  →データがないはずのUnnamed:9カラムに日付らしきものがある

        Args:
            data (pd.DataFrame): csvから読み出したデータ

        Returns:
            pd.DataFrame: 引数のデータからUnnamedカラムを削除処理されたもの
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

    def delete_nan_row(self, data: pd.DataFrame) -> pd.DataFrame:
        """すべてのデータがnanの行を削除する
           例:日付~栄養を示すデータがすべてnanの場合、削除
              group   日付  献立名 材料名 分量（g）  エネルギー（kcal）  たんぱく質（g）  脂質（g）  ナトリウム（mg）


        Args:
            data (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = data.copy()
        data.dropna(how='all', axis=0, inplace=True)
        return data

    def data_formatting(self, data: pd.DataFrame) -> pd.DataFrame:
        formatting_data = data.copy()
        formatting_data = self.delete_nan_row(formatting_data)
        formatting_data = self.separate_column_name(formatting_data)
        formatting_data = self.delete_unname_columns(formatting_data)
        del data
        return formatting_data

    def data_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        formatting_data = self.data_formatting(data)
        return formatting_data
