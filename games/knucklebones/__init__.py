from base.registry import register_game
from games.knucklebones.game import Knucklebones
from games.knucklebones.node import KnucklebonesNode
from games.knucklebones.openloop_node import KnucklebonesOpenLoopNode

register_game("knucklebones", game_cls=Knucklebones, node_cls=KnucklebonesNode, dice=6)
register_game("knucklebones_open", game_cls=Knucklebones, node_cls=KnucklebonesOpenLoopNode, dice=6)
