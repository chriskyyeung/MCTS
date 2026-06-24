from base.registry import register_game
from games.tictactoe.game import TicTacToe
from games.tictactoe.node import TicTacToeNode
from games.tictactoe.trainer import TicTacToeTrainer

register_game("tictactoe", game_cls=TicTacToe, node_cls=TicTacToeNode, trainer_cls=TicTacToeTrainer)
