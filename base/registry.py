"""Central registry for game constructors and MCTS node types."""

from typing import Any

_GAME_REGISTRY: dict[str, dict[str, Any]] = {}


def register_game(name: str, *, game_cls, node_cls, trainer_cls=None, dice: int = -1, **kwargs):
    _GAME_REGISTRY[name] = {
        "game_cls": game_cls,
        "node_cls": node_cls,
        "trainer_cls": trainer_cls,
        "dice": dice,
        **kwargs,
    }


def get_game(name: str) -> dict:
    return _GAME_REGISTRY[name]


def list_games() -> list[str]:
    return list(_GAME_REGISTRY.keys())
