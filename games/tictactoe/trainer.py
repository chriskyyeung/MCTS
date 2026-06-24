import numpy as np
import torch

from base.nn.game_net import GameData
from base.nn.trainer import Trainer
from games.tictactoe.game import TicTacToe


class TicTacToeTrainer(Trainer):
    ROT_IDX = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    FLR_IDX = [2, 1, 0, 5, 4, 3, 8, 7, 6]

    def __init__(self, config_dir: str = "configs", use_cuda: bool = False) -> None:
        super().__init__("tictactoe", TicTacToe, config_dir, use_cuda)

    def generate_symmetry(self, board: np.ndarray, p: np.ndarray) -> np.ndarray:
        board_sym = [board]
        p_sym = [p]

        board_r = np.rot90(board)
        while np.any(board_r != board):
            board_sym.append(board_r)
            p_sym.append(p_sym[-1][self.ROT_IDX])

            board_r = np.rot90(board_r)

        board_f = np.fliplr(board)
        for i in range(len(board_sym)):
            if np.all(board_f == board_sym[i]):
                return np.stack(board_sym), np.stack(p_sym)

        for i in range(len(board_sym)):
            board_sym.append(np.fliplr(board_sym[i]))
            p_sym.append(p_sym[i][self.FLR_IDX])

        return np.stack(board_sym), np.stack(p_sym)


if __name__ == "__main__":
    from base.config import Config

    torch.serialization.add_safe_globals([GameData])

    game = "tictactoe"
    config = Config.load_game("configs", game)
    t = TicTacToeTrainer(use_cuda=True)
    train = True

    if train:
        # For repeat training
        for i in range(config["n_iteration"]):
            t.generate_battle_record(config, False)

            player = t.get_model(config["model_in_path"], config["game_net"])
            battle_record = torch.load(config["battle_path"], weights_only=True)
            print(f"Loaded {config['battle_path']}")
            print(f"Length of records = {len(battle_record)}")

            new_player = t.train(
                player,
                battle_record,
                config["n_epoch"],
                Config.load("configs/hyperparameter.yaml"),
            )

            new_player.dump(config["model_out_path"])

            config["battle_version"] += 1
            config["model_in_version"] += 1
            config["model_out_version"] += 1
            import yaml

            with open(f"configs/{game}.yaml", "w") as f:
                yaml.dump(config, f)
    else:
        players = []
        print(config["model_in_path"])
        players.append(t.get_model(config["model_in_path"], config["game_net"]))
        config["model_in_version"] = 8
        players.append(t.get_model(config["model_in_path"], config["game_net"]))
        t.vs_battle(players, 25)
