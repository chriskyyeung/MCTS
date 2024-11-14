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
from game import Game
from knucklebones.knucklebones import Knucklebones
from knucklebones.knucklebones_node import KnucklebonesNode
from tictactoe.tictactoe import TicTacToe
from tictactoe.tictactoe_node import TicTacToeNode

game: list[GameState] = [TicTacToe, Connect4, Knucklebones]
node: list[MCTSNode] = [TicTacToeNode, Connet4Node, KnucklebonesNode]
game_name: list[str] = ["tictactoe", "connect4", "knucklebones"]

def eve_simulation(game_id: int, game_config: dict, first_hand: int, second_hand: int, n_round: int, log_config: dict) -> tuple:
    msg = ""
    msg += "c = {c:5.1f}, n = {n_simulation:5d} vs ".format(**game_config[first_hand])
    msg += "c = {c:5.1f}, n = {n_simulation:5d}".format(**game_config[second_hand])
    logger.info(msg)

    game = Game()
    game.game_id = game_id
    game.config = dict()
    game.config["log_config"] = log_config

    result = [0, 0, 0]
    for _ in tqdm(range(n_round)):
        game.initialize_board()

        turn = 0
        while not game.board.is_game_over:
            if turn % 2 == 0:
                game.run_turn(game_config[first_hand], False)
            else:
                game.run_turn(game_config[second_hand], False)
            turn += 1

        result[int(game.board.game_result)] += 1
    
    logger.info(f"Result ({first_hand}, {second_hand}): {result}")
    return first_hand, second_hand, result
        
if __name__ == "__main__":
    with open("config.yaml") as config:
        config = yaml.safe_load(config)["EVE"]

    game_id = config["game_id"]
    game_config = []
    config["log_config"]["filename"] = f"EvE_{game_name[game_id]}.log"

    logging.basicConfig(**config["log_config"])
    logger = logging.getLogger("Main program")
    
    for c, t, n in product(*config["game_config"][game_name[game_id]].values()):
        game_config.append({"c": c, "simulation_time": t, "n_simulation": n})
    
    n_player = len(game_config)
    competition_record = [[None for _ in range(n_player) ] for _ in range(n_player)]

    match_up = [(game_id, game_config, i, j, config["n_round"], config["log_config"]) for i in range(n_player) for j in range(n_player)]
    with Pool(config["nproc"]) as pool:
        result = pool.starmap(eve_simulation, match_up)
    
    for i, j, count in result:
        competition_record[i][j] = count
    
    np.save(f"record_{game_name[game_id]}", np.array(competition_record))