import pandas as pd

from analyzer.config import Config


class Preprocessing():
    """読み込みデータ一つに対して前処理を行うクラス
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def rename_column(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def data_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.rename_column(data)
        return data
