import numpy as np

from base.nn.trainer import Trainer
from games.connect4.game import Connect4


class Connect4Trainer(Trainer):
    def __init__(self, config_dir: str = "configs", use_cuda: bool = False) -> None:
        super().__init__("connect4", Connect4, config_dir, use_cuda)

    def generate_symmetry(self, board: np.ndarray, p: np.ndarray) -> tuple:
        """Connect4 has only left-right mirror symmetry."""
        board_flip = np.fliplr(board)
        if np.array_equal(board_flip, board):
            return board.reshape(1, *board.shape), p.reshape(1, *p.shape)

        p_flip = p[::-1]  # reverse column order
        return (
            np.stack([board, board_flip]),
            np.stack([p, p_flip]),
        )
