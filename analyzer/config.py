from typing import Dict


class Config():

    def __init__(self, config_json: Dict) -> None:
        """前処理の設定を変数に指定する

        Args:
            config_json (Dict): 前処理の設定を記したjsonファイルを読み込んだもの
        """
        self.config = config_json
        formating = self.config['formatting']
        manipulating = self.config['manipulating']

        self.analysis_target = self.config['analysis_target']

        self.rename_col = formating['rename_col']

        self.nan_col = manipulating['nan_col']
        self.nan_name = manipulating['nan_name']
        self.str_num_col = manipulating['str_num_col']
        self.datetime_col = manipulating['datetime_col']
