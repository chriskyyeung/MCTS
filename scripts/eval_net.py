import argparse
import ast
import sys
from pathlib import Path

# Add project root to path so we can import from 'base' and 'games'
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

import games  # noqa: F401
from base.config import Config
from base.nn.game_net import GameData, GameNet
from base.nn.puct_node import PUCTNode, PUCTRoot
from base.registry import get_game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="tictactoe", help="Game to evaluate")
    parser.add_argument("--model-version", type=int, default=None, help="Model version to load")
    args = parser.parse_args()
    game_name = args.game

    # Required for loading the model correctly
    torch.serialization.add_safe_globals([GameData])

    # Load configurations
    try:
        main_config = Config.load("configs/mcts.yaml", "main")
        net_config = Config.load_game("configs", game_name)
    except Exception as e:
        print(f"Failed to load configurations: {e}")
        return

    mcts_config = main_config["game_config"].get(game_name, {"c": 1, "simulation_time": 5, "n_simulation": 500})
    mcts_simulation_time = mcts_config["simulation_time"]
    mcts_n_simulation = mcts_config["n_simulation"]
    mcts_c = mcts_config["c"]

    net_n_simulation = net_config["n_simulation"]

    eval_version = args.model_version if args.model_version is not None else net_config.get("eval_model_version", 0)
    model_in_path = net_config["model_format"].format(game=game_name, version=eval_version)

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== {game_name.capitalize()} Network Evaluation ===")
    print(f"Device: {device}")
    print(f"Network config -> Simulations: {net_n_simulation}")
    print(f"Pure MCTS config -> Simulations: {mcts_n_simulation}, Time: {mcts_simulation_time}, c: {mcts_c}")

    # Load Network Model
    print(f"Loading network model from: {model_in_path}")
    game_net = GameNet(device=device, **net_config["game_net"])
    try:
        game_net.load(model_in_path)
        game_net.eval()
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    try:
        mode = ast.literal_eval(
            input(
                "Select mode:\n"
                "[1] EvE (Network vs Pure MCTS)\n"
                "[2] PvE (Human vs Network)\n"
                "[3] EvP (Network vs Human)\n"
                "Choice: "
            )
        )
    except Exception:
        print("Invalid input.")
        return

    # Assign players based on mode
    # [Player 1 (first hand), Player 2 (second hand)]
    if mode == 1:
        players = ["Net", "MCTS"]
    elif mode == 2:
        players = ["Human", "Net"]
    elif mode == 3:
        players = ["Net", "Human"]
    else:
        print("Invalid mode selected.")
        return

    print(f"\nMatchup: {players[0]} (1st) vs {players[1]} (2nd)")

    # Initialize game
    game_reg = get_game(game_name)
    game = game_reg["game_cls"]()
    game.print()

    # Initialize trees for AIs
    net_root = PUCTRoot(device)
    mcts_node = game_reg["node_cls"](game)
    current_net_node = PUCTNode(game, None, 0)

    trainer = game_reg["trainer_cls"](use_cuda=(device == "cuda")) if game_reg.get("trainer_cls") else None
    transform_fn = trainer.transform_board if trainer else lambda b, t: torch.from_numpy(b).float()

    turn = 0
    while not game.is_game_over:
        current_player = players[turn % 2]
        print(f"\n--- Turn {turn + 1}: {current_player}'s move ---")

        if current_player == "Human":
            action = game.prompt_next_move()
            game = game.update(action)

            # Recreate PUCTNode for Network
            current_net_node = PUCTNode(game, None, 0)

            # Update Pure MCTS node
            mcts_node = mcts_node.get_child_by_action(action)
            if not mcts_node:
                mcts_node = game_reg["node_cls"](game)
            else:
                mcts_node.state = game

        elif current_player == "MCTS":
            # Search and return the best child node
            mcts_node = mcts_node.best_action(
                c=mcts_c, simulation_time=mcts_simulation_time, n_simulation=mcts_n_simulation
            )
            action = mcts_node.parent_action
            game = game.update(action)

            # Recreate PUCTNode for Network based on the new game state
            current_net_node = PUCTNode(game, None, 0)

        elif current_player == "Net":
            # Search and return the best child node
            current_net_node = net_root.search(
                current_net_node, net_n_simulation, game_net, transform_fn, use_dirichlet=False
            )
            # parent_action is move_id for PUCTNode
            move_id = current_net_node.parent_action
            action = game.all_actions[move_id]
            game = game.update(action)

            # Update Pure MCTS node
            mcts_node = mcts_node.get_child_by_action(action)
            if not mcts_node:
                mcts_node = game_reg["node_cls"](game)
            else:
                mcts_node.state = game

        game.print()
        turn += 1

    # Print final result
    print("\n=== Game Over ===")
    if game.game_result == 1:
        print(f"{players[0]} (Player 1) wins!")
    elif game.game_result == -1:
        print(f"{players[1]} (Player 2) wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()
