from typing import Dict


class Config():

    def __init__(self, config_json: Dict) -> None:
        """前処理の設定項目config.jsonの値を変数に代入する

        Args:
            config_json (Dict): config.jsonを読み込んだ内容
        """
        self.config = config_json
        formating = self.config['formatting']
        manipulating = self.config['manipulating']

        self.analysis_target = self.config['analysis_target']

        self.rename_col = formating['rename_col']

        self.nan_name = manipulating['nan_name']
        self.str_num_col = manipulating['str_num_col']
        self.datetime_col = manipulating['datetime_col']
