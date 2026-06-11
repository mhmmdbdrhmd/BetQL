from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

ACTION_STAND = 0
ACTION_HIT = 1
ACTION_DOUBLE = 2
ACTION_SPLIT = 3
ACTION_NAMES = {ACTION_STAND: 'stand', ACTION_HIT: 'hit',
                ACTION_DOUBLE: 'double', ACTION_SPLIT: 'split'}

RULES = ('american', 'european')


class blackjack_env(Env):
    """Blackjack environment.

    Actions: 0 = stand, 1 = hit, 2 = double down, 3 = split.
    Doubling is offered on any first two cards (american) or only on a
    two-card hard 9, 10 or 11 (european); splitting only on a pair.
    Invalid double/split choices fall back to hit so the agent can
    always act from any state.

    Rules:
      - 'american': dealer is dealt a hole card up front. A dealer
        blackjack only costs the player the original bet (the peek
        would have ended the round before doubling/splitting).
      - 'european': dealer takes no hole card until the player has
        finished. A dealer blackjack collects the full (doubled) bet.
    A player natural (two-card 21, no split) pays 3:2.
    """

    def __init__(self, rules='american'):
        super(blackjack_env, self).__init__()
        if rules not in RULES:
            raise ValueError("rules must be one of {}".format(RULES))
        self.rules = rules
        self.action_space = Discrete(4)
        # 19 player scores (4-22, 22 = bust) + 10 dealer cards +
        # usable ace + can double + can split
        self.observation_space = Box(low=0, high=1, shape=(32,), dtype=np.int8)

    def one_hot_encode(self, value, n):
        vector = np.zeros(n)
        vector[value] = 1
        return vector

    def get_state(self, observation):
        score, dealer_card, usable_ace, can_double, can_split = observation
        player_vector = self.one_hot_encode(score - 4, 19)
        dealer_vector = self.one_hot_encode(dealer_card - 1, 10)
        flags = np.array([usable_ace, can_double, can_split])
        return np.concatenate([player_vector, dealer_vector, flags])

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
        if 1 in hand and score + 10 <= 21:
            return score + 10
        if score > 21:
            score = 22
        return score

    def current_hand(self):
        return self.hands[self.current]

    def can_double(self):
        hand = self.current_hand()
        if len(hand) != 2:
            return False
        if self.rules == 'american':
            return True  # American tables: double on any first two cards
        # European tables: only on a two-card hard 9, 10 or 11
        # (no soft hand scores 9-11, so checking the best score suffices)
        return self.calculate_score(hand) in (9, 10, 11)

    def can_split(self):
        hand = self.current_hand()
        return len(hand) == 2 and hand[0] == hand[1] and not self.split_done

    def valid_actions(self):
        actions = [ACTION_STAND, ACTION_HIT]
        if self.can_double():
            actions.append(ACTION_DOUBLE)
        if self.can_split():
            actions.append(ACTION_SPLIT)
        return actions

    def step(self, action):
        assert self.action_space.contains(action)

        if action == ACTION_SPLIT and not self.can_split():
            action = ACTION_HIT
        if action == ACTION_DOUBLE and not self.can_double():
            action = ACTION_HIT

        hand = self.current_hand()

        if action == ACTION_SPLIT:
            card = hand[0]
            self.hands = [[card, self.draw_card()], [card, self.draw_card()]]
            self.doubled = [False, False]
            self.split_done = True
            return self.get_state(self._get_obs()), 0, False, {}

        if action == ACTION_DOUBLE:
            self.doubled[self.current] = True
            hand.append(self.draw_card())
            hand_done = True
        elif action == ACTION_HIT:
            hand.append(self.draw_card())
            hand_done = self.calculate_score(hand) > 21
        else:  # stand
            hand_done = True

        if not hand_done:
            return self.get_state(self._get_obs()), 0, False, {}

        if self.current + 1 < len(self.hands):
            self.current += 1
            return self.get_state(self._get_obs()), 0, False, {}

        reward = self._resolve()
        return self.get_state(self._get_obs()), reward, True, {}

    def _resolve(self):
        while len(self.dealer_hand) < 2 or self.calculate_score(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())
        dealer_score = self.calculate_score(self.dealer_hand)
        dealer_natural = len(self.dealer_hand) == 2 and dealer_score == 21

        total = 0.0
        for hand, doubled in zip(self.hands, self.doubled):
            bet = 2 if doubled else 1
            score = self.calculate_score(hand)
            player_natural = (not self.split_done and not doubled
                              and len(hand) == 2 and score == 21)
            if score > 21:
                total -= bet
            elif dealer_natural and not player_natural:
                total -= bet if self.rules == 'european' else 1
            elif player_natural and dealer_natural:
                total += 0
            elif player_natural:
                total += 1.5
            elif dealer_score > 21 or score > dealer_score:
                total += bet
            elif score < dealer_score:
                total -= bet
        return total

    def _get_obs(self):
        hand = self.current_hand()
        score = self.calculate_score(hand)
        usable_ace = 1 if 1 in hand and sum(hand) + 10 <= 21 else 0
        return (score, self.dealer_hand[0], usable_ace,
                1 if self.can_double() else 0,
                1 if self.can_split() else 0)

    def reset(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.hands = [self.deal_hand()]
        self.doubled = [False]
        self.current = 0
        self.split_done = False
        if self.rules == 'european':
            self.dealer_hand = [self.draw_card()]
        else:
            self.dealer_hand = self.deal_hand()
        return self.get_state(self._get_obs())

    def render(self, display_full_dealer=False):
        for i, hand in enumerate(self.hands):
            marker = ''
            if len(self.hands) > 1:
                marker = " (hand {}{})".format(
                    i + 1, ", playing" if i == self.current and not display_full_dealer else "")
            doubled = " [doubled]" if self.doubled[i] else ""
            print("Player's hand{}: {} - score: {}{}".format(
                marker, hand, self.calculate_score(hand), doubled))
        if display_full_dealer:
            print("Dealer's hand: {} - score: {}".format(
                self.dealer_hand, self.calculate_score(self.dealer_hand)))
        elif self.rules == 'european':
            print("Dealer's hand: {} (no hole card) \n".format(self.dealer_hand[0]))
        else:
            print("Dealer's hand: {} and [hidden] \n".format(self.dealer_hand[0]))
