import logging
from itertools import product
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from base.registry import list_games
from scripts.play import Game

logger = logging.getLogger(__name__)

# Using registry dynamically inside the main block instead of hardcoding.


def eve_simulation(
    game_config_name: str,
    game_config: dict,
    first_hand: int,
    second_hand: int,
    n_round: int,
    log_config: dict,
) -> tuple:
    msg = ""
    msg += "c = {c:5.1f}, n = {n_simulation:5d} vs ".format(**game_config[first_hand])
    msg += "c = {c:5.1f}, n = {n_simulation:5d}".format(**game_config[second_hand])
    logger.info(msg)

    game = Game()
    game.game_name = game_config_name
    game.config = dict()
    game.config["log_config"] = log_config

    action = None
    result = [0, 0, 0]
    for _ in tqdm(range(n_round)):
        players = [[game_config[first_hand], None], [game_config[second_hand], None]]
        game.initialize_board()

        turn = 0
        while not game.board.is_game_over:
            if players[turn % 2][1]:
                players[turn % 2][1] = players[turn % 2][1].get_child_by_action(action)
                if players[turn % 2][1]:
                    players[turn % 2][1].state = game.board
            players[turn % 2][1], action = game.run_turn(*players[turn % 2], False)

            turn += 1

        result[int(game.board.game_result)] += 1

    logger.info(f"Result ({first_hand}, {second_hand}): {result}")
    return first_hand, second_hand, result


if __name__ == "__main__":
    from base.config import Config

    config = Config.load("configs/mcts.yaml", "EVE")
    game_id = config["game_id"]
    game_config = []

    available_games = list_games()
    game_name_str = available_games[game_id]

    config["log_config"]["filename"] = f"EvE_{game_name_str}.log"

    logging.basicConfig(**config["log_config"])
    logger = logging.getLogger("Main program")

    for c, t, n in product(*config["game_config"][game_name_str].values()):
        game_config.append({"c": c, "simulation_time": t, "n_simulation": n})

    n_player = len(game_config)
    competition_record = [[None for _ in range(n_player)] for _ in range(n_player)]

    match_up = [
        (game_name_str, game_config, i, j, config["n_round"], config["log_config"])
        for i in range(n_player)
        for j in range(n_player)
    ]
    with Pool(config["nproc"]) as pool:
        result = pool.starmap(eve_simulation, match_up)

    for i, j, count in result:
        competition_record[i][j] = count

    np.save(f"record_{game_name_str}", np.array(competition_record))
