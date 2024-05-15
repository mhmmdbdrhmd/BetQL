from agents.q_learning_agent import QLearningAgent
from environments.blackjack_env import blackjack_env
import os
import numpy as np

class Trainer:
    def __init__(self, env, agent):
        self.agent = agent
        self.env=env

    def train_agent(self, nb_steps=50000):
        # Ask user if they want to load existing weights
        load_weights = input("Do you want to load existing weights before training? (y/n): ").strip().lower()
        if load_weights == 'y':
            filename = input("Enter the filename of the weights to load: ")
            if os.path.exists("weights/"+ filename + ".h5f.index"):
                print(f"Loading weights from {filename}.h5f")
                self.agent.load("weights/"+ filename + ".h5f.index")
            else:
                print("File not found. Starting training with a new model.")
        
        self.agent.compile(lr=1e-3)
        self.agent.train(nb_steps=nb_steps)
        
        scores=self.agent.agent.test(self.env, nb_episodes=10000, visualize=False,verbose=0)
        print("\n Average reward earned by the agent on 10000 matches:{}\n".format(np.mean(scores.history['episode_reward'])))
        
        # Save the trained model
        filename = input("Please enter a filename to save the trained weights: ")
        if filename:
            self.agent.save("weights/"+ f"{filename}.h5f")
        else:
            self.agent.save("weights/default_weights.h5f")

def train():
    env = blackjack_env()
    agent = QLearningAgent(env=env, total_states=env.observation_space.shape[0], actions=env.action_space.n)
    trainer = Trainer(env=env, agent=agent)


    trainer.train_agent(nb_steps=50000)
    return env,agent  # To indicate training completed
