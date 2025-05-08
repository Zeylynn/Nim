import random

class Nim():
    def __init__(self, initial=[4, 4, 4, 4]):
        self.piles = initial.copy()
        self.player = 0  # Player 0 starts
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action
        self.piles[pile] -= count
        self.switch_player()
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player



class NimAI():
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()  # Q-value table
        self.q[(0, 0, 0, 2), (3, 2)] = -1 # Test Q-Value 
        self.q[(0, 0, 0, 2), (3, 1)] = 10 # Test Q-Value 

        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate

        self.update_q_value((2, 1, 1, 0), (0, 1), 0.2, 1, 0.8)
        print(self.q[(2, 1, 1, 0), (0, 1)])

    def update(self, old_state, action, new_state, reward):
        old_q = self.get_q_value(old_state, action)
        best_future_q = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, best_future_q)

    def get_q_value(self, state, action):
        try:
            return self.q[state, action]
        except:
            return 0

    def update_q_value(self, state, action, old_q, reward, future_q):
        self.q[state, action] = old_q + self.alpha * ((reward + future_q) - old_q)
    
    def best_future_reward(self, state):
        """
        Determine the highest Q-value among all possible actions in a given state.
        
        Parameters:
            state (list): The state for which to compute the best future reward.
            
        Returns:
            float: The highest Q-value among available actions. 
                Returns 0 if no actions are available.
        """
        raise NotImplementedError

    def choose_action(self, state, epsilon=True):
        """
    Choose an action for the given state using an epsilon-greedy strategy.
    
    Parameters:
        state (list): The current game state.
        epsilon (bool): If True, use epsilon-greedy exploration; otherwise, choose the best action.
    
    Returns:
        tuple: The chosen action from the available actions.
    """
        raise NotImplementedError

def train(n):
    player = NimAI()

    for i in range(n):
        game = Nim([4, 4, 4, 4])
        last_move = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}

        while True:
            state = game.piles.copy()
            action = player.choose_action(state)
            last_move[game.player]["state"] = state
            last_move[game.player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(last_move[game.player]["state"], last_move[game.player]["action"], new_state, 1)
                break
            elif last_move[game.player]["state"] is not None:
                player.update(last_move[game.player]["state"], last_move[game.player]["action"], new_state, 0)

    return player

player = NimAI()