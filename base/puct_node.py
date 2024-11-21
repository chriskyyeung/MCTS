from math import sqrt
from typing import Self, Any

import numpy as np
import torch

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
    def N(self) -> int:
        # Retrieve the value from self.parent
        return self.parent.N_a[self.parent_action] if self.parent else 0

    @N.setter
    def N(self, value):
        # Parent stores the visit of each child
        # print(self.parent_action, value)
        # self.state.print()
        self.parent.N_a[self.parent_action] = value
        # print(self.parent.N_a)
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
        return self.c_puct * sqrt(self.N) * self.p_a  / (1 + self.N_a)
    
    def no_legal_move(self):
        # Default: Game ends when no legal moves. Thus do nothing
        return

    def best_child(self) -> Self:
        best_moves = (self.Q_a + self.U_a)[self.state.legal_index]
        return self.state.legal_index[np.random.choice(np.where(best_moves == np.max(best_moves))[0])]
        
    def expand(self, p_a) -> Self:
        """_summary_

        Args:
            p_a (_type_): _description_

        Returns:
            Self: _description_
        """
        if not self.state.is_game_over and self.state.legal_index:
            self.is_expanded = True
            mask = np.repeat(True, len(p_a))
            mask[self.state.legal_index] = False
            p_a[mask] = 0.
            p_a /= np.sum(p_a)
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
                    current.state.update(current.state.all_actions[move_id]),
                    current,
                    move_id,
                )
            
            current = current.child[move_id]
            # current.state.print()
    
        return current
    
    def backpropagate(self, score: Any) -> None:    
        """Backpropagation, triggering parent's one if any

        Args:
            score (int): Score of the current state
        """
        if self.parent:
            self.N += 1
            self.V += score
            self.parent.backpropagate(score*-1)
        return

class PUCTRoot:
    """The root of the tree of PUCT roots and perform the search
    """
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.parent = None # Must be None
        self.child: GameState = None # Only one child which is the current state
        self.N_a = [0]
        self.V_a = [0]
        pass

    def set_child(self, new_child: PUCTNode):
        # Point to the new child
        self.child = new_child
        self.N_a[0] = new_child.N
        self.V_a[0] = new_child.V

        # Change the parent of the child
        new_child.parent = self
        new_child.parent_action = 0
        return

    def backpropagate(self, *args) -> None:
        # To terminate the backpropagate
        return

    def transform_board_for_torch(self, board_non_torch: Any):
        """Game specific transformation for preparing input for its net

        Args:
            board_non_torch (Any): The default board for the game

        Returns:
            torch.: _description_
        """
        # Default is to treat it as 1 batch, 1 channel 
        board = torch.from_numpy(board_non_torch)
        board = board.view(1, 1, *board.shape).to(self.device).float()
        return board

    def search(self, current: PUCTNode, n_search: int, game_net: torch.nn.Module):
        self.set_child(current)

        with torch.no_grad():
            for _ in range(n_search):
                # Selection until reaching a leaf
                leaf = self.child.select()

                # Use the net to get p, v pair
                board = self.transform_board_for_torch(leaf.state._board)
                p_a, v_a = game_net(board)
                p_a = torch.nn.Softmax(dim=1)(p_a).flatten().detach().cpu().numpy()
                v_a = v_a.item()
            
                # Expansion if needed
                if not (leaf.is_terminal or leaf.is_expanded):
                    leaf.expand(p_a)
                
                # Backpropagation
                leaf.backpropagate(v_a)
        
        # Only consider number of visit for result
        max_visit = np.where(self.child.N_a == np.max(self.child.N_a))[0]
        move_id = np.random.choice(max_visit)
        if move_id not in self.child.child:
            self.child.child[move_id] = PUCTNode(
                self.child.state.update(self.child.state.all_actions[move_id]),
                self.child,
                move_id,
            )
        
        # Return result
        return self.child.child[move_id]

    
if __name__ == "__main__":
    from base.config import Config
    from game_net import GameNet    
    from tictactoe.tictactoe import TicTacToe

    root = PUCTRoot(False)
    config = Config.load("game_net.yaml", "tictactoe")["game_net"]
    random_net = GameNet(**config)

    current = TicTacToe()
    current.print()
    while not current.is_game_over:
        a, current = root.search(current, 25, random_net)
        print(a)
        # current = current.update(current.all_actions[a])
        current = current.child[a]
        current.print()

    # empty_state = PUCTNode(TicTacToe(), None, None)
    # for _ in range(10):
    #     current = empty_state
    #     for _ in range(5):
    #         if current.is_terminal:
    #             break

    #         if not current.is_expanded:
    #             current.expand(np.random.rand(9))
    #         print(current.p_a)
    #         print(current.Q_a)
    #         print(current.U_a)

    #         current = current.select()
        
    #     current.backpropagate(1)
    #     print(empty_state.N_a)
    #     print(empty_state.V_a)
    #     print("------")
