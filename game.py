from functools import partial
import numpy as np
import yaml

from base.game_state import GameState
from base.mcts_node import MCTSNode
from base.non_determ_mcts_node import NonDetermMCTSNode
from connect4.connect4 import Connect4
from connect4.connect4_node import Connet4Node
from knucklebones.knucklebones import Knucklebones
from knucklebones.knucklebones_node import KnucklebonesNode
from knucklebones.knucklebones_openloop_node import KnucklebonesOpenLoopNode
from tictactoe.tictactoe import TicTacToe
from tictactoe.tictactoe_node import TicTacToeNode

class Game:
    def __init__(self) -> None:
        self.game: list[GameState] = [TicTacToe, Connect4, Knucklebones, partial(Knucklebones,False)]
        self.node: list[MCTSNode] = [TicTacToeNode, Connet4Node, KnucklebonesNode, KnucklebonesOpenLoopNode]
        self.game_name: list[str] = ["tictactoe", "connect4", "knucklebones", "knucklebones"]
        self.dice: list[int] = [-1, -1, 6, 6]
        pass

    def load_config(self, section) -> None:
        with open("config.yaml") as config:
            self.config = yaml.safe_load(config)[section]
        return
    
    def game_mode_selection(self):
        # Load config
        self.load_config("main")

        # Ask for the game mode to be run
        self.game_id = eval(input("Select game ([1] Tic-tac-toe / [2] Connect-4 / [3] Knucklebones): ")) - 1
        self.game_mode = eval(input("Select mode ([1] PVE / [2] EVP / [3] EVE): "))

        self.initialize_board()
        self.board.print()
        return
    
    def initialize_board(self):
        # Intialize the board
        self.board: GameState = self.game[self.game_id]()
        return

    def start(self):
        istep = 1 if self.game_mode == 2 else 0
        while not self.board.is_game_over:
            if (self.game_mode < 3) and (istep % 2) == 0:
                self.run_turn()
            else:
                self.run_turn(self.config["game_config"][self.game_name[self.game_id]])

            istep += 1

    def run_turn(self, ai_config:dict = None, is_print: bool = True):
        n = self.dice[self.game_id]
        if n >= 0:
            n = self.roll_a_dice(n)
            if is_print:
                print(f"The rolled dice = {n}")

        if ai_config:
            action = self.computer_move(
                dice = n,
                ai_config=ai_config,
            )
        else:
            if n > 0:
                action = (n, self.board.prompt_next_move())
            else:
                action = self.board.prompt_next_move()

        self.board = self.board.update(action)
        if is_print:
            self.board.print()
        return


    def computer_move(
        self,
        ai_config: dict,
        dice: int = -1,
    ) -> tuple:
        # Intialise the MCTS node
        if dice >= 0:
            discrete_states = np.zeros(6)
            discrete_states[dice-1] = 1.
            computer_node: NonDetermMCTSNode = self.node[self.game_id](
                self.board,
                discrete_states=discrete_states,
                log_config=self.config["log_config"]
            )
        else:
            computer_node: MCTSNode = self.node[self.game_id](
                self.board,
                log_config=self.config["log_config"]
            )

        # Get the best action
        best_child = computer_node.best_action(**ai_config)
        return best_child.parent_action
    
    @staticmethod
    def roll_a_dice(n=6):
        return np.random.randint(n) + 1

if __name__ == "__main__":
    game = Game()
    game.game_mode_selection()
    game.start()