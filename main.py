import logging

from base.game_state import GameState
from base.mcts_node import MCTSNode
from connect4.connect4 import Connect4
from connect4.connect4_node import Connet4Node
from tictactoe.tictactoe import TicTacToe
from tictactoe.tictactoe_node import TicTacToeNode


log_config = {
    # "level": logging.DEBUG,
    "level": logging.INFO,
    "format": '%(asctime)s [%(levelname)s] :\n%(message)s',
}

game: list[GameState] = [TicTacToe, Connect4]
node: list[MCTSNode] = [TicTacToeNode, Connet4Node]
game_config = [{"c":0.2, "n_simulation": 5000}, {"c":0.5, "n_simulation": 2000}, {"c":1, "n_simulation": 300}]

def computer_move(board: GameState, node: MCTSNode, c=1, n_simulation=1000) -> tuple:
    node = node(board, log_config=log_config)
    best_child = node.best_action(c, n_simulation)
    return best_child.parent_action

def transform(config, sim):
    config['c'] = 0.1 / config['c']
    config['n_simulation'] = sim - config["n_simulation"]

if __name__ == "__main__":
    game_id = eval(input("Select game ([1] Tic-tac-toe / [2] Connect-4): ")) - 1
    game_mode = input("Select mode ([1] PVE / [2] EVE): ")

    board: GameState = game[game_id]()
    board.print()
    sim = game_config[game_id]["n_simulation"] * 2 #10
    
    istep = 0
    while not board.is_game_over:
        if (game_mode == "1") and (istep % 2) == 0:
            action = board.prompt_next_move()
        else:
            # if (game_mode == "2"):
            #     transform(game_config[game_id], sim)
            #     print(game_config[game_id])
            action = computer_move(board, node[game_id], **game_config[game_id])
        board = board.update(action)
        board.print()
        istep += 1
