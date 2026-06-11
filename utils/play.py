from environments.blackjack_env import blackjack_env, ACTION_NAMES
from utils.model_store import load_model, choose_model

GREEN = '\033[92m'
RESET = '\033[0m'


class Player:
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent

    def _ask_action(self, observation):
        valid = self.env.valid_actions()
        suggested = None
        if self.agent:
            suggested = self.agent.suggest(observation)
            if suggested not in valid:
                suggested = 1  # the env falls back to hit for invalid choices
        labels = []
        for action in valid:
            name = ACTION_NAMES[action]
            labels.append(GREEN + name + RESET if action == suggested else name)
        name_to_action = {ACTION_NAMES[a]: a for a in valid}
        while True:
            choice = input("Enter {}: ".format(' / '.join("'{}'".format(l) for l in labels))).strip().lower()
            if choice in name_to_action:
                return name_to_action[choice]
            print("Invalid choice.")

    def play(self):
        while True:
            observation = self.env.reset()
            done = False
            reward = 0

            while not done:
                self.env.render()
                action = self._ask_action(observation)
                observation, reward, done, info = self.env.step(action)
                self.env.render(display_full_dealer=done)

            print("Game over. ",
                  "You win {:g}!".format(reward) if reward > 0
                  else "You lose {:g}!".format(-reward) if reward < 0
                  else "It's a draw!")

            if input("Play again? (y/n): ").strip().lower() != 'y':
                break
        return True


def play(env=None, rl_agent=None):
    if rl_agent is None:
        if input("Do you want to load an agent for suggestions? (y/n): ").strip().lower() == 'y':
            name = choose_model()
            if name:
                env, rl_agent = load_model(name)
            else:
                print("Continuing without an agent.")

    if env is None:
        env = blackjack_env()

    player = Player(env, agent=rl_agent)
    player.play()
    return True
