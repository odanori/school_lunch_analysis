from typing import Dict


class Config():

    def __init__(self, config_json: Dict) -> None:
        """_summary_

        Args:
            config_json (Dict): _description_
        """
        self.config = config_json
        formating = self.config['formatting']
        manipulating = self.config['manipulating']

        self.analysis_target = self.config['analysis_target']

        self.rename_col = formating['rename_col']

        self.nan_col = manipulating['nan_col']
        self.nan_name = manipulating['nan_name']
