from copy import deepcopy
from typing import Self, Any

import numpy as np

from base.game_state import GameState

class Knucklebones(GameState):
    """Class for the state of a Connect-4 game
    """
    _dice = np.array(range(7), dtype=int)
    _token = ["_"] + [i for i in range(1,7)]

    def __init__(self) -> None:
        super().__init__()

        self._board = np.zeros((2,3,3), dtype=int)
        self._moveID = 0
        self._count = np.zeros(7, dtype=int)

        self._layer_status = np.zeros(2, dtype=int) # check if the board is full
        self._row_score   = np.zeros((2,3), dtype=int) # Row scores
        self._nxt_row      = np.zeros((2,3), dtype=int)


    @property
    def is_game_over(self) -> bool:
        """Whether the game ends

        Returns:
            bool: Does the game end
        """
        return np.any(self._layer_status == 9)

    @property
    def game_result(self) -> int:
        """Result of the game and should be called at the game end

        Returns:
            int : Score of the current game
                  - 1 means first-player wins, 
                  - -1 means second-player wins
                  - and 0 means draw
        """
        player_score = np.sum(self._row_score, axis=1)
        if player_score[0] > player_score[1]:
            return 1
        elif player_score[0] < player_score[1]:
            return -1
        else:
            return 0

    def _is_valid_move(self, irow: int, dice: int) -> bool:
        """Check if the input row is invalid/full

        Args:
            irow (int): Which row would the dice be placed
            dice (int): The value of the dice

        Returns:
            bool: Whether the move is valid
        """
        return (0 < dice < 7) and (irow < 3) and (self._nxt_row[self._moveID, irow] < 3)

    @staticmethod
    def _get_row_score(row: np.ndarray) -> int:
        """Return the score of the input row

        Args:
            row (np.ndarray): Array with values fomr 0 to 6 inclusively

        Returns:
            int: Score of the row
        """
        occurence = np.zeros(7, dtype=int)
        for v in row:
            occurence[v] += 1
        return np.dot(Knucklebones._dice, np.power(occurence, 2))

    def _tidy_up_row(self, layer: int, irow: int, cnt: int) -> None:
        """Update the score and status of the specified layer

        Args:
            layer (int): Layer index, 0/1
            irow (int): Row index, 0-2
            cnt (int): No. of row entries added(+ve) / removed(-ve)
        """
        self._layer_status[layer] += cnt
        self._nxt_row[layer, irow] += cnt
        self._row_score[layer, irow] = self._get_row_score(self._board[layer, irow, :])
        return

    def _update_status(self, action: tuple[int, int]) -> None:
        """Update the status based on the current game board

        Args:
            action (tuple[int, int]): (irow, dice)
        """
        irow, this_move = action
        self._tidy_up_row(self._moveID, irow, 1)

        # Check if any removal on opponent's layer
        layer, idx = 1 - self._moveID, 0
        new_row = [0] * 3
        for n in self._board[layer, irow, :]:
            if n != this_move:
                new_row[idx] = n
                idx += 1
        
        # Update the opponent board and its status
        self._board[layer, irow, :] = new_row
        self._tidy_up_row(layer, irow, idx-3)

        # Now, it's opponent's turn
        self._moveID = layer
        return

    def _move(self, action: tuple[int, int]) -> np.ndarray:
        """Make the input move on the board and update the game status

        Args:
            action (tuple[int, int]): (irow, dice)

        Returns:
            np.ndarray: The new game board
        """
        irow, dice = action
        assert self._is_valid_move(irow, dice)

        self._board[self._moveID, irow, self._nxt_row[self._moveID, irow]] = dice
        self._update_status(action)
        return self._board

    def get_legal_actions(self) -> list:
        """To return legal actions of the current game state

        Raises:
            NotImplementedError: Legal action generation is a must

        Returns:
            list: All legal actions on position, without the dice value
        """
        return np.where(self._nxt_row[self._moveID, :] < 3)[0].tolist()

    @staticmethod
    def _get_row_output(row: np.ndarray) -> str:
        """Print a single row

        Args:
            row (np.ndarray): Row to be printed, which should be a single row
        
        Returns:
            str: Formatted string recording the row status
        """
        grid = [ "|{t}".format(t=Knucklebones._token[i]) for i in row]
        grid = "".join(grid) + "|"
        return grid

    def print(self) -> None:
        """Print the game board
        """
        for irow in range(self._board.shape[1]):
            row = (
                self._get_row_output(self._board[0, irow, ::-1])
                + " {:02d}|{:02d} ".format(*self._row_score[:,irow])
                + self._get_row_output(self._board[1, irow, :])
            )
            print(row)
        print()
    
    @staticmethod
    def prompt_next_move() -> None:
        """Get the move from the human player

        Returns:
            int: Row index
        """
        return eval(input("Choose the row to place the dice (i-th row): "))


if __name__ == "__main__":
    kb = Knucklebones()
    moves = [(2,2),(0,2),(2,3),(1,6),(2,3),(1,6),(0,1),(2,3)]

    for m in moves:
        kb = kb.update(m)
        kb.print()