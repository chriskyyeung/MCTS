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
    def game_net(cls, config, board_shape):
        """Derive network dimensions from board_shape.

        Args:
            config: The game_net section of the config dict.
            board_shape: Tuple of (batch, channel, height, width).
        """
        config["conv_config"]["in_channel"] = board_shape[1]
        config["residual_config"]["in_channel"] = config["conv_config"]["out_channel"]
        config["policy_value_config"]["in_channel"] = config["residual_config"]["in_channel"]
        board_area = board_shape[2] * board_shape[3]
        config["policy_value_config"]["board_area"] = board_area
        # Default action_space = board_area; games can override in YAML
        config["policy_value_config"].setdefault("action_space", board_area)
        return config

    @classmethod
    def load_game(cls, config_dir: str, game_name: str) -> dict:
        """Load a game-specific config and resolve paths.

        Args:
            config_dir: Directory containing per-game YAML config files.
            game_name: Name of the game (matches YAML filename).

        Returns:
            dict: Fully resolved configuration dictionary.
        """
        config = cls.load(str(Path(config_dir) / f"{game_name}.yaml"))
        if "board_shape" in config:
            config["game_net"] = cls.game_net(config["game_net"], config["board_shape"])

        config = cls.update_config(config, game_name)
        return config

    @classmethod
    def update_config(cls, config: dict, game_name: str) -> dict:
        config["battle_path"] = config["battle_format"].format(game=game_name, version=config["battle_version"])
        config["model_in_path"] = config["model_format"].format(game=game_name, version=config["model_in_version"])
        config["model_out_path"] = config["model_format"].format(game=game_name, version=config["model_out_version"])
        return config


if __name__ == "__main__":
    print(Config.load_game("configs", "tictactoe"))
