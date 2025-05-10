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
        """
        sonst bekomm ich einen Fehler dass None-Objects nicht entpackt werden können
        aber nachdem ich die anderen Funktionen üebrarbeitet hab gehts
        if action == None:
            self.switch_player()
            self.winner = self.player
        """
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
        self.q[tuple(state), action] = old_q + self.alpha * ((reward + future_q) - old_q)
    
    def best_future_reward(self, state):
        highest_q_value = -10000    # der Q-Wert vom besten Zug
        possible_actions = []       # die liste mit allen möglichen Zügen für den übergebenen State
        # geht alle möglichen actions durch und gibt sie in eine Liste
        for i in range(len(state)):
            pile = state[i]
            
            for j in range(pile, 0, -1):
                possible_actions.append((i, j))

        # prüft alle möglichen actions auf ihren Q-Wert, der größte Q-Wert wird zurückgegeben
        # wenn keine Action possible gibt 0 zurück
        if len(possible_actions) != 0:
            for action in possible_actions:
                value = self.get_q_value(state, action)
                if value > highest_q_value:
                    highest_q_value = value
            return highest_q_value
        return 0

    def choose_action(self, state, epsilon=True):
        all_actions = []

        for key, _ in self.q.items():
            if state in key:    # example key ((2,2,2,2), (1,1))
                all_actions.append(key[1])

        #  wenn epsilon = 0.8, dann sind 80 von 100 werten kleienr => also 80% Exploration
        if (len(all_actions) == 0) or ((random.randint(0, 100) / 100) < self.epsilon and epsilon):
            # gleiches Prinzip wie oben
            possible_actions = []
            for i in range(len(state)):
                pile = state[i]
                
                for j in range(pile, 0, -1):
                    possible_actions.append((i, j))

            if len(possible_actions) == 0:
                return None
            return random.choice(possible_actions)
        else:
            best_move = self.best_future_reward(state)
            for action in all_actions:
                if self.q[(state), (action)] == best_move:
                    return action

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