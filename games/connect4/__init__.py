from base.registry import register_game
from games.connect4.game import Connect4
from games.connect4.node import Connect4Node
from games.connect4.trainer import Connect4Trainer

register_game("connect4", game_cls=Connect4, node_cls=Connect4Node, trainer_cls=Connect4Trainer)
