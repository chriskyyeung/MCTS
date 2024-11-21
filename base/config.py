from pathlib import Path

import yaml

class Config:
    def __init__(self) -> None:
        pass

    @classmethod
    def load(cls, file_path: str, target: str = None, **kwargs):
        assert Path(file_path).is_file(), "Invalid path"

        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        
        if target:
            if hasattr(cls, target):
                return getattr(cls, target)(config[target], **kwargs)
            else:
                return config[target]
        
        return config

    @classmethod
    def game_net(cls, config, size):
        config["conv_config"]["in_channel"] = size[-1]
        config["residual_config"]["in_channel"] = config["conv_config"]["out_channel"]
        config["policy_value_config"]["in_channel"] = config["residual_config"]["in_channel"]
        config["policy_value_config"]["height"] = size[1]
        config["policy_value_config"]["width"] = size[2]
        config["policy_value_config"]["p_output"] = size[1] * size[2]
        return config

    @classmethod
    def tictactoe(cls, config):
        config["battle_path"] = config["battle_path"].format(game="tictactoe", version=config["battle_version"])
        config["model_in_path"] = config["model_in_path"].format(game="tictactoe", version=config["model_in_version"])
        config["model_out_path"] = config["model_out_path"].format(game="tictactoe", version=config["model_out_version"])
        config["game_net"] = cls.game_net(config["game_net"], (1,3,3,1))
        return config

if __name__ == "__main__":
    print(Config.load("config.yaml", "EVE"))
    print(Config.load("config.yaml", "main"))