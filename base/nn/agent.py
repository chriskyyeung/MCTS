"""Protocol for NN-based game agents, covering both perfect and imperfect info."""

from typing import Protocol, runtime_checkable

import torch

from base.game_state import GameState


@runtime_checkable
class NNAgent(Protocol):
    """Interface that all NN-based agents must satisfy."""

    def select_action(self, state: GameState) -> int:
        """Given a game state, return the action index to play."""
        ...

    def get_model(self) -> torch.nn.Module:
        """Return the underlying neural network."""
        ...
