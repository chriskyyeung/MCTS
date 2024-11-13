import yaml

import numpy as np

from base.game_state import GameState
from base.mcts_node import MCTSNode
from base.non_determ_mcts_node import NonDetermMCTSNode
from connect4.connect4 import Connect4
from connect4.connect4_node import Connet4Node
from knucklebones.knucklebones import Knucklebones
from knucklebones.knucklebones_node import KnucklebonesNode
from tictactoe.tictactoe import TicTacToe
from tictactoe.tictactoe_node import TicTacToeNode

game: list[GameState] = [TicTacToe, Connect4, Knucklebones]
node: list[MCTSNode | NonDetermMCTSNode] = [TicTacToeNode, Connet4Node, KnucklebonesNode]
game_config: list[dict] = [
    {"c":1, "n_simulation": 100, "simulation_time": 5}, # tic-tac-toe
    {"c":1, "n_simulation": float("inf"), "simulation_time": 10}, # connect4
    {"c":1, "n_simulation": 30000, "simulation_time": 10}, # Knucklebones
]

def computer_move(
        board: GameState, 
        node: MCTSNode | NonDetermMCTSNode,
        log_config: dict,
        c: float=1,
        n_simulation: int=1,
        simulation_time: float=1000,
        dice: int = -1,
    ) -> tuple:
    
    # Intialise the MCTS node
    if dice >= 0:
        discrete_states = np.zeros(6)
        discrete_states[dice-1] = 1.
        computer_node: NonDetermMCTSNode = node(board, discrete_states=discrete_states, log_config=log_config)
    else:
        computer_node: MCTSNode = node(board, log_config=log_config)

    # Get the best action
    best_child = computer_node.best_action(c=c, n_simulation=n_simulation, simulation_time=simulation_time)

    return best_child.parent_action

def roll_a_dice():
    return np.random.randint(6) + 1

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)["main"]
    
    game_list = ["tictactoe", "connect4", "Knucklebones"]
    game_id = eval(input("Select game ([1] Tic-tac-toe / [2] Connect-4 / [3] Knucklebones): ")) - 1
    game_mode = eval(input("Select mode ([1] PVE / [2] EVP / [3] EVE): "))

    
    board: GameState = game[game_id]()
    board.print()
    
    istep = 1 if game_mode == 2 else 0
    while not board.is_game_over:
        n = -1
        if game_id == 2:
            n = roll_a_dice()
            print(f"The rolled dice = {n}")

        if (game_mode < 3) and (istep % 2) == 0:
            if game_id == 2:
                action = (board.prompt_next_move(), n)
            else:
                action = board.prompt_next_move()
        else:
            action = computer_move(
                board,
                node[game_id],
                config["log_config"],
                dice = n,
                **config["game_config"][game_list[game_id]]
            )

        board = board.update(action)
        board.print()
        istep += 1
