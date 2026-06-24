import argparse
import sys
from pathlib import Path

# Add project root to path so we can import from 'base' and 'games'
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

import games  # noqa: F401
from base.config import Config
from base.nn.game_net import GameData
from base.registry import get_game


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero MCTS Network")
    parser.add_argument("--game", type=str, default="", help="Game to train (e.g. tictactoe, connect4)")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--eval-only", action="store_true", help="Run vs_battle instead of training")
    args = parser.parse_args()

    game_name = args.game
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.serialization.add_safe_globals([GameData])

    try:
        config = Config.load_game("configs", game_name)
    except Exception as e:
        print(f"Failed to load configurations: {e}, game: {game_name}")
        return

    game_reg = get_game(game_name)
    trainer_cls = game_reg.get("trainer_cls")
    if not trainer_cls:
        print(f"No trainer registered for game '{game_name}'")
        return

    t = trainer_cls(use_cuda=use_cuda)

    if not args.eval_only:
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
    else:
        players = []
        print(config["model_in_path"])
        players.append(t.get_model(config["model_in_path"], config["game_net"]))
        config["model_in_version"] = 8
        players.append(t.get_model(config["model_in_path"], config["game_net"]))
        t.vs_battle(players, 25)


if __name__ == "__main__":
    main()
