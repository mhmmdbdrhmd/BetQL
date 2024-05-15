from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
class blackjack_env(Env):
    def __init__(self):
        super(blackjack_env, self).__init__()
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        
        self.action_space = Discrete(2)  # 0 = stand, 1 = hit
        self.observation_space = Box(low=0, high=1, shape=(31,), dtype=np.int8)
        tf.get_logger().setLevel('ERROR')

    

        
    def one_hot_encode(self,value, n):
        """One-hot encode a value given the number of possible values."""
        vector = np.zeros(n)
        vector[value] = 1
        return vector

    def get_state(self,observation):
        """Convert the tuple of discrete observations into a one-hot encoded state."""
        player_score, dealer_card, usable_ace = observation
        player_vector = self.one_hot_encode(player_score - 4 , 19)  # 19 possible player scores (4-22)
        dealer_vector = self.one_hot_encode(dealer_card - 1, 10)  # 10 possible dealer showing cards (1-10)
        ace_vector = self.one_hot_encode(usable_ace, 2)  # Usable ace can be 0 or 1
        state = np.concatenate([player_vector, dealer_vector, ace_vector])
        return state
    
    def draw_card(self):
        if len(self.deck) == 0:
            self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        card = np.random.choice(self.deck)
        self.deck.remove(card)
        return card

    def deal_hand(self):
        return [self.draw_card(), self.draw_card()]

    def calculate_score(self, hand):
        score = sum(hand)
        # Adjust for aces
        if 1 in hand and score + 10 <= 21:
            return score + 10
        if score > 21 :
            score = 22 
        return score

    def step(self, action):
        
        assert self.action_space.contains(action)
        
        
        if action == 1:  # hit
            self.player_hand.append(self.draw_card())
            if self.calculate_score(self.player_hand) > 21:
                done = True
                reward = -1  # Lose the bet
            else:
                done = False
                reward = 0
        else:  # stand
            done = True
            while self.calculate_score(self.dealer_hand) < 17:
                self.dealer_hand.append(self.draw_card())
            dealer_score = self.calculate_score(self.dealer_hand)
            player_score = self.calculate_score(self.player_hand)
            if dealer_score > 21 or player_score > dealer_score:
                reward = 1  # Win the bet
            elif player_score < dealer_score:
                reward = -1  # Lose the bet
            else:
                reward = 0  # Draw

        return self.get_state(self._get_obs()), reward, done, {}


    def _get_obs(self):
        player_score = self.calculate_score(self.player_hand)
        return (player_score-4, self.dealer_hand[0], 1 if 1 in self.player_hand and player_score + 10 <= 21 else 0)

    def reset(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.player_hand = self.deal_hand()
        self.dealer_hand = self.deal_hand()
        return self.get_state(self._get_obs())

    def render(self,  display_full_dealer=False):
        """Prints the current game state. Displays full dealer hand if game is over."""
        print(f"Player's hand: {self.player_hand} - score: {self.calculate_score(self.player_hand)}")
        if display_full_dealer:
            print(f"Dealer's hand: {self.dealer_hand} - score: {self.calculate_score(self.dealer_hand)}")
        else:
            print(f"Dealer's hand: {self.dealer_hand[0]} and [hidden] \n")

