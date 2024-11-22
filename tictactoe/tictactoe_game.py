import numpy as np

from base.game_state import GameState

class TicTacToe(GameState):
    def __init__(self) -> None:
        """To construct the game state in the beginning
        """
        # Basic board setting
        self._board = np.zeros((3,3))
        self._turnID = 1

        # Game status check
        self._row_status = np.zeros(3, dtype=int)
        self._col_status = np.zeros(3, dtype=int)
        self._diag_status = np.zeros(2, dtype=int)

        # Call parent class to initialize action
        super().__init__()
        pass

    def check_game_over(self) -> bool:
        """Whether the game ends

        Returns:
            bool: Does the game end?
        """
        # Check 1) draw, 2) row, 3) column, 4) diagonal
        self.is_game_over = (
            np.all(self._board != 0) \
            or np.any(np.abs(self._row_status) == 3) \
            or np.any(np.abs(self._col_status) == 3) \
            or np.any(np.abs(self._diag_status) == 3)
        )
        return

    @property
    def game_result(self) -> int:
        """Result of the game and should be called at the game end

        Returns:
            Int : Score of the current game.
                  - 1 means first-player wins, 
                  - -1 means second-player wins
                  - and 0 means draw
        """
        # Check row, col and diagonal
        for r in *self._row_status, *self._col_status, *self._diag_status:
            if np.abs(r) == 3:
                return r // 3
        return 0

    def _is_valid_move(self, action: tuple[int, int]) -> bool:
        """Check if the input coordinate is a valid move

        Args:
            i (int): x-coordinate of the move (0-2)
            j (int): y-coordinate of the move (0-2)

        Returns:
            bool: Whether the move is valid
        """
        i, j = action
        return self._board[i,j] == 0

    def initialize_actions(self) -> list:
        return [(i,j) for i in range(3) for j in range(3)]

    def _update_status(self, action: tuple[int, int]) -> None:
        """Update the status variables after a move

        Args:
            action (tuple[int, int]): Move that has been made
        """
        i, j = action
        self._row_status[i] += self._turnID
        self._col_status[j] += self._turnID

        # left diagonal
        if i == j:
            self._diag_status[0] += self._turnID
        
        # right diagonal
        if i + j == 2:
            self._diag_status[1] += self._turnID

        # It's opposite turn now
        self._turnID *= -1
        return
    
    def _move(self, action: tuple[int, int]) -> np.ndarray:
        """Make the input move on the board and update the game status

        Args:
            action (tuple[int, int]): Cooridnate of the desired move

        Returns:
            np.ndarray: The new game board
        """
        assert self._is_valid_move(action)

        i, j = action
        board = self._board.copy()
        board[i, j] = self._turnID

        self._update_status(action)
        return board
    
    def print(self) -> None:
        """To print the game board
        """
        print(self._board)
        print()
        return

    @staticmethod
    def prompt_next_move() -> tuple[int, int]:
        """Get the move from the human player

        Returns:
            tuple[int, int]: The input move
        """
        return eval(input("Enter your move (i,j): "))