from itertools import product
from multiprocessing import Pool
import logging

import numpy as np
from tqdm import tqdm
import yaml

from base.game_state import GameState
from base.mcts_node import MCTSNode
from connect4.connect4 import Connect4
from connect4.connect4_node import Connet4Node
from knucklebones.knucklebones import Knucklebones
from knucklebones.knucklebones_node import KnucklebonesNode
from tictactoe.tictactoe import TicTacToe
from tictactoe.tictactoe_node import TicTacToeNode

game: list[GameState] = [TicTacToe, Connect4, Knucklebones]
node: list[MCTSNode] = [TicTacToeNode, Connet4Node, KnucklebonesNode]
game_name: list[str] = ["tictactoe", "connect4", "knucklebones"]

def load_config():
    with open("config.yaml") as config:
        yaml_config = yaml.safe_load(config)["EVE"]
    return yaml_config

def computer_move(
        board: GameState,
        node: MCTSNode,
        log_config: dict,
        c: int = 1,
        n_simulation:int = 1,
        simulation_time: float= 10,
    )-> tuple:
    node = node(board, log_config=log_config)
    best_child = node.best_action(c=c, n_simulation=n_simulation, simulation_time=simulation_time)
    return best_child.parent_action

def eve_simulation(game_config: dict, i: int, j: int, n_round: int, log_config: dict) -> tuple:
    player1, player2 = game_config[i], game_config[j]
    msg = ""
    msg += "c = {c:5.1f}, n = {n_simulation:5d} vs ".format(**player1)
    msg += "c = {c:5.1f}, n = {n_simulation:5d}".format(**player2)
    logger.info(msg)

    result = [0, 0, 0]
    for _ in tqdm(range(n_round)):
        turn = 0
        board: GameState = game[game_id]()

        while not board.is_game_over:
            if turn % 2 == 0:
                action = computer_move(board, node[game_id], log_config, **player1)
            else:
                action = computer_move(board, node[game_id], log_config, **player2)
            board = board.update(action)

            turn += 1

        result[int(board.game_result)] += 1
    
    logger.info(f"Result ({i}, {j}): {result}")
    return i, j, result
        
if __name__ == "__main__":
    config = load_config()

    game_id = config["game_id"]
    game_config = []
    for c, t, n in product(*config["game_config"][game_name[game_id]].values()):
        game_config.append({"c": c, "simulation_time": t, "n_simulation": n})
    
    n_player = len(game_config)
    competition_record = [[None for _ in range(n_player) ] for _ in range(n_player)]


    config["log_config"]["filename"] = f"EvE_{game_name[game_id]}.log"
    logging.basicConfig(**config["log_config"])
    logger = logging.getLogger("Main program")

    match_up = [(game_config, i, j, config["n_round"], config["log_config"]) for i in range(n_player) for j in range(n_player)]
    with Pool(config["nproc"]) as pool:
        result = pool.starmap(eve_simulation, match_up)
    
    for i, j, count in result:
        competition_record[i][j] = count
    
    np.save(f"record_{game_name[game_id]}", np.array(competition_record))