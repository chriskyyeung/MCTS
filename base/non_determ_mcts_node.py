from typing import Self, Any

import numpy as np

from base.game_state import GameState
from base.mcts_node import MCTSNode

class NonDetermMCTSNode(MCTSNode):
    """Base class for a MCTS node for non-deterministic game
    """
    def __init__(
            self, 
            state: GameState, 
            parent: Self = None,
            parent_action: Any = None,
            discrete_states: np.ndarray = np.array([1.]),
            log_config: dict = dict(),
        ) -> None:
        """Construct a node to perform MCTS on the input game state

        Args:
            state (GameState): 
                Current game state
            parent (Self, optional): 
                Parent node of the current state. 
                Defaults to None.
            parent_action (Any, optional):
                Action taken by the parent node to reach here. 
                Defaults to None.
            discrete_states (np.ndarray, optional):
                Probability for each discrete sta
                tes. 
                Defaults to np.array([1.]).
            log_config (dict, optional): 
                Configuration for `logging` library. 
                Defaults to dict().
        """
        super().__init__(state, parent, parent_action, log_config)

        # Handling random state
        self._set_random_state(discrete_states)
        
        # Currently designed for numeric "action" values
        self._id_to_move = list(set(s[1] for s in self._untried_actions))
        self._move_to_id = {move: i for i, move in enumerate(self._id_to_move)}
        self.children = [[None] * len(self._id_to_move) for _ in range(len(discrete_states))]
        pass

    @property
    def child_weights(self) -> list:
        weights = np.repeat(0, repeats=len(self.children[0]))
        for irow in range(len(self.children[0])):
            node_weight = 0
            node_N = 0

            for istate in range(len(self.children)):
                if self.children[istate][irow] is not None:
                    node_weight += self.children[istate][irow]._W
                    node_N += self.children[istate][irow]._N
        
            if node_N > 0:
                weights[irow] = node_weight / node_N + self._c * np.sqrt(2 * np.log(self._N)/ node_N)
    
        return weights

    @property
    def _random_state(self) -> Any:
        """Return a random state based on the random_state_config"""
        raise NotImplementedError("")

    def get_child_by_action(self, action) -> Self:
        for child in self.children[action[0]-1]:
            if child.parent_action == action:
                return child
        return None

    def _set_random_state(self, discrete_states) -> None:
        # Handling random state
        self._random_state_config = {
            'a': range(len(discrete_states)),
            'p': discrete_states,
        }
        return

    def _get_child_weights_with_state(self, dice: int) -> np.ndarray:
        weights = -np.inf
        for inode, node in enumerate(self.children[dice]):
            if node is not None and node._N > 0:
                w = node._W / node._N + self._c * np.sqrt(2 * np.log(self._N)/ node._N)
                if w > weights:
                    weights = w
                    max_node = [inode]
                elif w == weights:
                    max_node.append(inode)
        return np.random.choice(max_node)

    def _get_action(self) -> list:
        return self._untried_actions.pop()
    
    def best_child(self) -> Self:
        dice = self._random_state
        irow = self._get_child_weights_with_state(dice-1)
        return self.children[dice-1][irow]

    def update_node_N(self) -> None:
        self._N = 0
        for i, p in enumerate(self._random_state_config["p"]):
            if p > 0:
                self._N += np.sum(c._N for c in self.children[i] if c)
        return

    def _get_random_state_index(self, size: int) -> int:
        return np.random.choice(size=size, **self._random_state_config)[0]

    def expand(self) -> Self:
        """Append an expanded node as children and return it

        Returns:
            Self: The expanded child node
        """
        action = self._get_action()
        action_id = self._move_to_id[action[1]]
        self.children[action[0]-1][action_id] =self.__class__(
                self.state.update(action),
                parent=self,
                parent_action=action,
                discrete_states=np.repeat(1/6, repeats=6),
        )
        return self.children[action[0]-1][action_id]

    def selection(self, c: float) -> Self:
        """Selection step

        Args:
            c (float): Exploration parameter

        Returns:
            Self: Selected node
        """
        self._c = c
        current_node = self
        while (current_node.is_fully_expanded) and (not current_node.is_terminal):
            current_node = current_node.best_child()
            current_node._c = c

        return current_node

    @staticmethod
    def simulation_step(state: GameState, random_state: Any) -> Any:
        """Simulation 1 step forward from the input game state

        Args:
            state (GameState): Current game state
            random_state (Any): Generated random state

        Returns:
            Any: 1 simulated move forward from the current game state
        """
        possible_moves = [s for s in state.get_legal_actions() if s[0] == random_state]
        assigned_move = np.random.randint(len(possible_moves))
        return possible_moves[assigned_move]

    def simulation(self) -> tuple[GameState, Any]:
        """Simulation, involving multiple simulation steps

        Returns:
            - tuple[GameState, Any]: A tuple containing
                1. Final state of the simulated game
                2. Result, in terms of the score
        """
        current_state = self.state
        while not current_state.is_game_over:
            action = self.simulation_step(current_state, self._random_state)
            current_state = current_state.update(action)
        
        return current_state, current_state.game_result