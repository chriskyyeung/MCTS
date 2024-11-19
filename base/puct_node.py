from math import sqrt
from typing import Self, Any

import numpy as np

from base.game_state import GameState

class PUCTNode:
    """This class is a MCTS with neural network
    """
    def __init__(self, state: GameState, parent: Self, parent_action: Any) -> None:
        self.state = state
        self.n_action = len(self.state.all_actions)
        self.is_expanded = False

        self.c_puct = 1.
        self.parent = parent
        self.parent_action = parent_action
        self.child = dict()
        self.V_a = np.zeros(self.n_action, dtype=np.float32) # Value of each child, for calculation of Q_a
        self.p_a = np.zeros(self.n_action, dtype=np.float32) # prior prob. of each child
        self.N_a = np.zeros(self.n_action, dtype=np.float32) # number of visit of each child
        pass

    @property
    def is_terminal(self) -> bool:
        # Check if the simulation ends
        return self.state.is_game_over

    @property
    def n_visit(self) -> int:
        # Retrieve the value from self.parent
        return self.parent.N_a[self.parent_action] if self.parent else 0

    @n_visit.setter
    def n_visit(self, value):
        # Parent stores the visit of each child
        self.parent.N_a[self.parent_action] = value
        pass

    @property
    def V(self) -> int:
        # Retrieve the value of this action from self.parent
        return self.parent.V_a[self.parent_action] if self.parent else 0

    @V.setter
    def V(self, value):
        # Parent stores the value of each child
        self.parent.V_a[self.parent_action] = value
        pass

    @property
    def Q_a(self):
        return self.V_a / (1 + self.N_a)
    
    @property
    def U_a(self):
        return self.c_puct * sqrt(self.n_visit) * self.p_a  / (1 + self.N_a)
    
    def no_legal_move(self):
        # Default: Game ends when no legal moves. Thus do nothing
        return

    def best_child(self) -> Self:
        best_moves = (self.Q_a + self.U_a)[self.state.legal_index]
        return np.random.choice(np.where(best_moves == np.max(best_moves))[0])
        
    def expand(self, p_a) -> Self:
        """_summary_

        Args:
            p_a (_type_): _description_

        Returns:
            Self: _description_
        """
        if not self.state.is_game_over and self.state.legal_index:
            self.is_expanded = True
        else:
            # Case handling depending on game rules...
            self.no_legal_move()

        self.p_a = p_a
        return

    def select(self):
        current = self
        while current.is_expanded:
            move_id = current.best_child()

            if move_id not in current.child:
                current.child[move_id] = PUCTNode(
                    current.state.update(current.state.all_actions[current.state.legal_index[move_id]]),
                    self,
                    move_id,
                )
            
            current = current.child[move_id]
            current.state.print()
    
        return current
    
    def backpropagate(self, score: Any) -> None:    
        """Backpropagation, triggering parent's one if any

        Args:
            score (int): Score of the current state
        """
        if self.parent:
            self.n_visit += 1
            self.V += score
            self.parent.backpropagate(score*-1)
        return 
    
if __name__ == "__main__":
    from tictactoe.tictactoe import TicTacToe

    root = PUCTNode(TicTacToe(), None, None)
    for _ in range(10):
        current = root
        for _ in range(5):
            if current.is_terminal:
                break

            if not current.is_expanded:
                current.expand(np.random.rand(9))
            print(current.p_a)
            print(current.Q_a)
            print(current.U_a)

            current = current.select()
        
        current.backpropagate(1)
        print(root.N_a)
        print(root.V_a)
        print("------")
