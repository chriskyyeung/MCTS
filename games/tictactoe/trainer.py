import numpy as np

from base.nn.trainer import Trainer
from games.tictactoe.game import TicTacToe


class TicTacToeTrainer(Trainer):
    ROT_IDX = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    FLR_IDX = [2, 1, 0, 5, 4, 3, 8, 7, 6]

    def __init__(self, config_dir: str = "configs", use_cuda: bool = False) -> None:
        super().__init__("tictactoe", TicTacToe, config_dir, use_cuda)

    def generate_symmetry(self, board: np.ndarray, p: np.ndarray) -> np.ndarray:
        board_sym = [board]
        p_sym = [p]

        board_r = np.rot90(board)
        while np.any(board_r != board):
            board_sym.append(board_r)
            p_sym.append(p_sym[-1][self.ROT_IDX])

            board_r = np.rot90(board_r)

        board_f = np.fliplr(board)
        for i in range(len(board_sym)):
            if np.all(board_f == board_sym[i]):
                return np.stack(board_sym), np.stack(p_sym)

        for i in range(len(board_sym)):
            board_sym.append(np.fliplr(board_sym[i]))
            p_sym.append(p_sym[i][self.FLR_IDX])

        return np.stack(board_sym), np.stack(p_sym)
