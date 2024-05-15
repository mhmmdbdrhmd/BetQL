import gym
from gym import spaces
import numpy as np

class BlackjackEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BlackjackEnv, self).__init__()
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.action_space = spaces.Discrete(2)  # Actions: 0 = stand, 1 = hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Player's hand total (1-31)
            spaces.Discrete(11),  # Dealer's showing card (1-10)
            spaces.Discrete(2)    # Player's usable ace (0 or 1)
        ))

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
        return score

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit
            self.player_hand.append(self.draw_card())
            if self.calculate_score(self.player_hand) > 21:
                done = True
                reward = -1  # Player busts
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
                reward = 1  # Player wins
            elif player_score < dealer_score:
                reward = -1  # Dealer wins
            else:
                reward = 0  # Draw

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        player_score = self.calculate_score(self.player_hand)
        return (player_score, self.dealer_hand[0], int(1 in self.player_hand and player_score + 10 <= 21))

    def reset(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.player_hand = self.deal_hand()
        self.dealer_hand = self.deal_hand()
        return self._get_obs()

    def render(self, mode='human', display_full_dealer=False):
        """Prints the current game state. Displays full dealer hand if game is over."""
        print(f"Player's hand: {self.player_hand} - score: {self.calculate_score(self.player_hand)}")
        if display_full_dealer:
            print(f"Dealer's hand: {self.dealer_hand} - score: {self.calculate_score(self.dealer_hand)}")
        else:
            print(f"Dealer's hand: {self.dealer_hand[0]} and [hidden]")

    def close(self):
        pass
