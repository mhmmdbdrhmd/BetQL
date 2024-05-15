from environments.blackjack_env import blackjack_env
import os

class Player:
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent

    def play(self):
        while True:
            observation = self.env.reset()
            done = False
            
            while not done:
                self.env.render()
                if self.agent:
                    suggested_action = self.agent.agent.forward(observation)
                    action = input("Enter \033[92m'hit'\033[0m or 'stand': ").strip().lower() if suggested_action == 1 else input("Enter 'hit' or \033[92m'stand'\033[0m: ").strip().lower()
                    action = 1 if action == 'hit' else 0

                else:
                    action = input("Enter 'hit' or 'stand': ").strip().lower()
                    action = 1 if action == 'hit' else 0
                
                observation, reward, done, info = self.env.step(action)
                self.env.render(display_full_dealer=done)
            
            print("Game over. ", "You win!" if reward > 0 else "You lose!" if reward < 0 else "It's a draw!")
            
            if not self.agent:
                play_again = input("Play again? (y/n): ").strip().lower()
                if play_again != 'y':
                    break
        return True  # To indicate game play completed

def play(env=None, rl_agent=None):
    if env is None:
        env = blackjack_env()

    if rl_agent is None:
        load_agent = input("Do you want to load an agent for suggestions? (y/n): ").strip().lower()
        if load_agent == 'y':
            filename = input("Enter the filename of the weights to load: ")
            if os.path.exists("weights/" +filename + ".h5f.index"):
                from agents.q_learning_agent import QLearningAgent  # Ensure this import doesn't cause a circular dependency
                rl_agent = QLearningAgent(env=env, total_states=31, actions=2)
                rl_agent.load("weights/" +filename + ".h5f.index")
            else:
                print("File not found. Continuing without an agent.")
    

    player = Player(env, agent=rl_agent)
    player.play()
    return True  # Return to main menu
