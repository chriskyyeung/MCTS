import numpy as np

from base.game_state import GameState

class Connect4(GameState):
    """Class for the state of a Connect-4 game
    """
    _token = ["_", chr(9679), chr(9675)] # empty slots and occupied slots
    _boundary = chr(8254) * 15 # boundary between board layers
    _index_row = "|" + "|".join([str(i) for i in range(7)]) + "|" # easy reference to index

    def __init__(self) -> None:
        """To construct the game state in the beginning
        """
        # Basic board setting
        self._turnID = 1
        self._board = np.zeros((6,7), dtype=int)

        # Game status check
        self._next = np.zeros(7, dtype=int)
        self._row_status = np.zeros((6,4), dtype=int)
        self._col_status = np.zeros((3,7), dtype=int)
        self._leftDiag_status = np.zeros((3,4), dtype=int)
        self._rightDiag_status = np.zeros((3,4), dtype=int)

        # Call parent class to initialize action
        super().__init__()
        pass

    @property
    def status(self) -> tuple[int, ...]:
        """Return all status checks for looping or other operations

        Returns:
            tuple[int, ...]: All status check numbers
        """
        return (
            *self._row_status.flatten(), 
            *self._col_status.flatten(),
            *self._leftDiag_status.flatten(),
            *self._rightDiag_status.flatten(),
        )

    def check_game_over(self) -> None:
        """Whether the game ends

        Returns:
            bool: Does the game end?
        """
        # Check draw
        if np.all(self._board != 0):
            self.is_game_over = True
            return
        
        # Check row, col, left diag, right diag
        for status in self.status:
            if (np.abs(status) == 4):
                self.is_game_over = True
                return

        return

    @property
    def game_result(self) -> int:
        """Result of the game and should be called at the game end

        Returns:
            int: Score of the current game.
                  - 1 means first-player wins, 
                  - -1 means second-player wins
                  - and 0 means draw
        """
        # Check if any consecutive 4
        for status in self.status:
            if np.abs(status) == 4:
                return status // 4
        
        return 0

    def initialize_actions(self) -> list:
        return [i for i in range(7)]

    def _is_valid_move(self, j: int) -> bool:
        """Check if the input column is invalid or full

        Args:
            j (int): Column index

        Returns:
            bool: Whether the move is valid
        """
        return (j < 7) and (self._next[j] < 6)

    def _update_status(self, j: int) -> None:
        """Update the status variables after a move

        Args:
            j (int): Move that has been made
        """
        # Update row status
        min_j, max_j = max(0, j-3), min(3, j) + 1
        self._row_status[self._next[j], min_j:max_j] += self._turnID

        # Update col status
        min_i, max_i = max(0, self._next[j]-3), min(2, self._next[j]) + 1
        self._col_status[min_i:max_i, j] += self._turnID
        
        # Update left diag status
        x, y = self._leftDiag_status.shape
        for diff in range(4):
            row, col = self._next[j] - diff, j + diff - 3
            if 0 <= row < x and 0 <= col < y:
                self._leftDiag_status[row, col] += self._turnID

        # Update right diag status
        for diff in range(4):
            row, col = self._next[j] - diff, j - diff
            if 0 <= row < x and 0 <= col < y:
                self._rightDiag_status[row, col] += self._turnID

        # Update turnID and _next at the very last
        self._turnID *= -1
        self._next[j] += 1
        return

    def _move(self, j: int) -> np.ndarray:
        """Make the input move on the board and update the game status

        Args:
            j (int): Column index

        Returns:
            np.ndarray: The new game board
        """
        assert self._is_valid_move(j)

        board = self._board.copy()
        board[self._next[j], j] = self._turnID

        self._update_status(j)
        return board
    
    @staticmethod
    def _get_row_output(row: np.ndarray) -> str:
        """Print a single row

        Args:
            row (np.ndarray): Row to be printed, which should be a single row
        
        Returns:
            str: Formatted string recording the row status
        """
        grid = [ "|{t}".format(t=Connect4._token[i]) for i in row]
        grid = "".join(grid) + "|"
        return grid

    def print(self) -> None:
        """Print the game board
        """
        for row in self._board[::-1,:]:
            print(self._get_row_output(row))
        print(Connect4._index_row)
        print()

    @staticmethod
    def prompt_next_move() -> int:
        """Get the move from the human player

        Returns:
            int: Column index
        """
        return eval(input("Enter your move (j-th column): "))

if __name__ == "__main__":
    c4 = Connect4()
    moves = [2,6,3,4,2,2,3,4,7]

    for m in moves:
        c4 = c4.update(m)
        c4.print()