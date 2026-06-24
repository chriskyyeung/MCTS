import ast

import numpy as np

from base.config import Config
from base.mcts_node import MCTSNode
from base.registry import get_game, list_games


class Game:
    def __init__(self, config_path: str = "configs/mcts.yaml") -> None:
        self.config_path: str = config_path
        # The game selection and available games are now driven by the registry.
        self.available_games = list_games()
        pass

    def game_mode_selection(self):
        # Load config
        self.config = Config.load(self.config_path, "main")

        # Ask for the game mode to be run
        game_list_str = " / ".join([f"[{i+1}] {name}" for i, name in enumerate(self.available_games)])
        self.game_id = ast.literal_eval(input(f"Select game ({game_list_str}): ")) - 1
        self.game_mode = ast.literal_eval(input("Select mode ([1] PVE / [2] EVP / [3] EVE): "))
        self.game_name = self.available_games[self.game_id]

        self.initialize_board()
        self.board.print()
        return

    def initialize_board(self):
        # Intialize the board
        game_reg = get_game(self.game_name)
        self.board = game_reg["game_cls"]()
        return

    def start(self):
        # Players configuration. None = human player
        players = [None, None]
        action = None

        if self.game_mode == 3:
            players[0] = [
                self.config["game_config"][self.game_name],
                None,
            ]
            players[1] = [
                self.config["game_config"][self.game_name],
                None,
            ]
        else:
            players[2 - self.game_mode] = [
                self.config["game_config"][self.game_name],
                None,
            ]

        istep = 0
        while not self.board.is_game_over:
            if players[istep % 2]:
                # Non empty configuration means this's a AI player
                if players[istep % 2][1]:
                    players[istep % 2][1] = players[istep % 2][1].get_child_by_action(action)
                    if players[istep % 2][1]:
                        players[istep % 2][1].state = self.board
                players[istep % 2][1], action = self.run_turn(*players[istep % 2])
            else:
                _, action = self.run_turn()

            istep += 1

    def run_turn(
        self,
        ai_config: dict = None,
        computer_node: MCTSNode = None,
        is_print: bool = True,
    ) -> tuple:
        game_reg = get_game(self.game_name)
        n = game_reg.get("dice", -1)
        if n >= 0:
            n = self.roll_a_dice(n)
            if is_print:
                print(f"The rolled dice = {n}")

        if ai_config:
            best_child = self.computer_move(
                dice=n,
                ai_config=ai_config,
                computer_node=computer_node,
            )
            action = best_child.parent_action
        else:
            best_child = None
            if n > 0:
                action = (n, self.board.prompt_next_move())
            else:
                action = self.board.prompt_next_move()

        self.board = self.board.update(action)
        if is_print:
            self.board.print()
        return best_child, action

    def computer_move(
        self,
        ai_config: dict,
        computer_node: MCTSNode = None,
        dice: int = -1,
    ) -> MCTSNode:
        game_reg = get_game(self.game_name)
        # Intialise the MCTS node
        if dice >= 0:
            discrete_states = np.zeros(6)
            discrete_states[dice - 1] = 1.0
            if computer_node:
                computer_node._set_random_state(discrete_states)
                computer_node.update_node_N()
            else:
                computer_node = game_reg["node_cls"](
                    self.board,
                    discrete_states=discrete_states,
                    log_config=self.config["log_config"],
                )
        else:
            if not computer_node:
                computer_node = game_reg["node_cls"](self.board, log_config=self.config["log_config"])

        # Return the best child
        return computer_node.best_action(**ai_config)

    @staticmethod
    def roll_a_dice(n=6):
        return np.random.randint(n) + 1


if __name__ == "__main__":
    game = Game()
    game.game_mode_selection()
    game.start()
