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
game_config = [{"c":0.1, "simulation_time": 5}]

def computer_move(board: GameState, node: NonDetermMCTSNode, dice: int, c=1, simulation_time=1000) -> tuple:
    discrete_states = np.zeros(6)
    discrete_states[dice-1] = 1.
    node = node(board, discrete_states=discrete_states, log_config=log_config)
    best_child = node.best_action(c, simulation_time)
    return best_child.parent_action

def transform(config, sim):
    config['c'] = 0.1 / config['c']
    config['simulation_time'] = sim - config["simulation_time"]

def row_a_dice():
    return np.random.randint(6) + 1

if __name__ == "__main__":
    game_id = 0
    game_mode = input("Select mode ([1] PVE / [2] EVE): ")

    board: GameState = game[game_id]()
    board.print()
    istep = 0
    while not board.is_game_over:
        n = row_a_dice()
        print(f"The rowed dice = {n}")
        if (game_mode == "1") and (istep % 2) == 0:
            action = (board.prompt_next_move(), n)
        else:
            action = computer_move(board, node[game_id], dice = n, **game_config[game_id])
        board = board.update(action)
        board.print()
        istep += 1