import ast

import torch

from base.config import Config
from base.game_net import GameData, GameNet
from base.puct_node import PUCTNode, PUCTRoot
from tictactoe.tictactoe_game import TicTacToe
from tictactoe.tictactoe_node import TicTacToeNode


def main():
    # Required for loading the model correctly
    torch.serialization.add_safe_globals([GameData])

    # Load configurations
    try:
        main_config = Config.load("config.yaml", "main")
        net_config = Config.load("game_net.yaml", "tictactoe")
    except Exception as e:
        print(f"Failed to load configurations: {e}")
        return

    mcts_config = main_config["game_config"]["tictactoe"]
    mcts_simulation_time = mcts_config["simulation_time"]
    mcts_n_simulation = mcts_config["n_simulation"]
    mcts_c = mcts_config["c"]

    net_n_simulation = net_config["n_simulation"]

    eval_version = net_config.get("eval_model_version", 0)
    model_in_path = net_config["model_format"].format(game="tictactoe", version=eval_version)

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Tic-Tac-Toe Network Evaluation ===")
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
    game = TicTacToe()
    game.print()

    # Initialize trees for AIs
    net_root = PUCTRoot(device)
    mcts_node = TicTacToeNode(game)
    current_net_node = PUCTNode(game, None, 0)

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
                mcts_node = TicTacToeNode(game)
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
            current_net_node = net_root.search(current_net_node, net_n_simulation, game_net, use_dirichlet=False)
            # parent_action is move_id for PUCTNode
            move_id = current_net_node.parent_action
            action = game.all_actions[move_id]
            game = game.update(action)

            # Update Pure MCTS node
            mcts_node = mcts_node.get_child_by_action(action)
            if not mcts_node:
                mcts_node = TicTacToeNode(game)
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
