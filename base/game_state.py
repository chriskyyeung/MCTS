"""This is the base class for recording game state
"""

from copy import deepcopy
from typing import Self, Any

class GameState:
    def __init__(self) -> None:
        pass

    def __deepcopy__(self, memo) -> Self:
        """To retrieve a deepcopy of the current GameState object

        Args:
            memo (dict): Used by copy.deepcopy

        Returns:
            GameState: A deep copy of the original GameState
        """
        cls = self.__class__
        self_copy = cls.__new__(cls)
        memo[id(self)] = self_copy
        for k, v in self.__dict__.items():
            setattr(self_copy, k, deepcopy(v, memo))
        return self_copy

    @property
    def is_game_over(self) -> bool:
        """Whether the game ends

        Returns:
            bool: Does the game end
        """
        raise NotImplementedError("Game over determination has to be defined")

    @property
    def game_result(self) -> Any:
        """Result of the game and should be called at the game end

        Returns:
            Any : score/reseult of the current game
        """
        raise NotImplementedError("Check the game state and return the score/result of the game")

    def _is_valid_move(self, *args) -> bool:
        raise NotImplementedError("Check if the move is compatible with the game state")

    def _move(self, action: Any) -> Any:
        """Make the input move on the board and update the game status

        Args:
            action (Any): Move to be implemented

        Raises:
            NotImplementedError: Please implement this function for updating a move

        Returns:
            Any: New game board situation is expected to be returned
        """
        raise NotImplementedError("Please implement this function for updating a move")

    def _get_all_actions(self) -> list:
        """To return all actions, regardless of its validity

        Returns:
            list: All actions
        """
        pass

    def get_legal_actions(self) -> list:
        """To return legal actions of the current game state

        Returns:
            list: All legal actions under the current game state
        """
        raise NotImplementedError("Generation of legal actions has to be defined")

    def update(self, action: Any) -> Self:
        """Update the game state and return a deepcopy of the state

        Args:
            action (int): Cooridnate of the desired move

        Returns:
            Self: Updated version on the deepcopy of the input state
        """
        temp = deepcopy(self)
        temp._board = temp._move(action)
        return temp

    def print(self) -> None:
        """To define a way to print the current state of the board
        """
        raise NotImplementedError("Define and print the game state")
    
    @staticmethod
    def prompt_next_move() -> None:
        """To prompt the human player to input the next move
        """
        raise NotImplementedError("Prompting function has to be defined")