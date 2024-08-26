import logging

import numpy as np

from base.game_state import GameState
from base.non_determ_mcts_node import NonDetermMCTSNode
from knucklebones.knucklebones import Knucklebones
from knucklebones.knucklebones_node import KnucklebonesNode


log_config = {
    # "level": logging.DEBUG,
    "level": logging.INFO,
    "format": '%(asctime)s [%(levelname)s] :\n%(message)s',
}

game: list[GameState] = [Knucklebones]
node: list[NonDetermMCTSNode] = [KnucklebonesNode]
game_config = [{"c":1, "n_simulation": 500}]

def computer_move(board: GameState, node: NonDetermMCTSNode, c=1, n_simulation=1000) -> tuple:
    node = node(board, log_config=log_config)
    best_child = node.best_action(c, n_simulation)
    return best_child.parent_action

def transform(config, sim):
    config['c'] = 0.1 / config['c']
    config['n_simulation'] = sim - config["n_simulation"]

def row_a_dice():
    return np.random.randint(6) + 1

if __name__ == "__main__":
    game_id = 0
    game_mode = input("Select mode ([1] PVE / [2] EVE): ")

    board: GameState = game[game_id]()
    board.print()
    sim = game_config[game_id]["n_simulation"] * 2 #10
    
    istep = 0
    while not board.is_game_over:
        n = row_a_dice()
        print(f"The rowed dice = {n}")
        if (game_mode == "1") and (istep % 2) == 0:
            action = board.prompt_next_move()
        else:
            # if (game_mode == "2"):
            #     transform(game_config[game_id], sim)
            #     print(game_config[game_id])
            action = computer_move(board, node[game_id], **game_config[game_id])
        action = (action, n)
        board = board.update(action)
        board.print()
        istep += 1
