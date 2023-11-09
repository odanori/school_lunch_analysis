from typing import Dict


class Config():

    def __init__(self, config_json: Dict) -> None:
        """_summary_

        Args:
            config_json (Dict): _description_
        """
        self.config = config_json
        self.analysis_target = self.config['analysis_target']
