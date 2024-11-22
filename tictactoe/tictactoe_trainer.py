import torch

from base.game_net import GameData, GameNet
from base.trainer import Trainer
from tictactoe.tictactoe_game import TicTacToe

class TicTacToeTrainer(Trainer):
    def __init__(self, game_config: str, use_cuda) -> None:
        super().__init__("tictactoe", TicTacToe, game_config, (1,3,3,1), use_cuda)
    
if __name__ == "__main__":
    from base.config import Config
    torch.serialization.add_safe_globals([GameData])

    game = "tictactoe"
    config = Config.load("game_net.yaml", game)
    t = TicTacToeTrainer("tictactoe", use_cuda=True)
    
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
            Config.load("game_net.yaml", "hyperparameter"),
        )

        new_player.dump(config["model_out_path"])

        config["battle_version"] += 1
        config["model_in_version"] += 1
        config["model_out_version"] += 1
        Config.tictactoe(config)