from agents.q_learning_agent import QLearningAgent, DEFAULT_HIDDEN_LAYERS, DEFAULT_DROPOUT
from environments.blackjack_env import blackjack_env, RULES
from utils.model_store import save_model, load_model, choose_model
import numpy as np


def _ask_rules():
    choice = input("Choose blackjack rules ({}) [american]: ".format('/'.join(RULES))).strip().lower()
    return choice if choice in RULES else 'american'


def _ask_int(prompt, default):
    value = input("{} [{}]: ".format(prompt, default)).strip()
    return int(value) if value.isdigit() else default


def _ask_float(prompt, default):
    value = input("{} [{}]: ".format(prompt, default)).strip()
    try:
        return float(value)
    except ValueError:
        return default


def _ask_architecture():
    choice = input("Use the default model architecture {}? (y/n) [y]: ".format(
        DEFAULT_HIDDEN_LAYERS)).strip().lower()
    if choice != 'n':
        return DEFAULT_HIDDEN_LAYERS, DEFAULT_DROPOUT
    layers_raw = input("Enter hidden layer sizes, comma-separated (e.g. 256,128,64): ").strip()
    layers = [int(x) for x in layers_raw.split(',') if x.strip().isdigit() and int(x) > 0]
    if not layers:
        print("No valid layer sizes given, using the default architecture.")
        return DEFAULT_HIDDEN_LAYERS, DEFAULT_DROPOUT
    dropout = _ask_float("Dropout rate (0 disables dropout)", DEFAULT_DROPOUT)
    if not 0 <= dropout < 1:
        dropout = DEFAULT_DROPOUT
    return layers, dropout


def _build_new(env):
    layers, dropout = _ask_architecture()
    return QLearningAgent(env=env,
                          total_states=env.observation_space.shape[0],
                          actions=env.action_space.n,
                          hidden_layers=layers, dropout=dropout)


class Trainer:
    def __init__(self, env, agent):
        self.agent = agent
        self.env = env

    def train_agent(self, nb_steps=50000, lr=1e-3, resume=False):
        self.agent.compile(lr=lr)
        self.agent.train(nb_steps=nb_steps, resume=resume)

        scores = self.agent.agent.test(self.env, nb_episodes=10000, visualize=False, verbose=0)
        print("\n Average reward earned by the agent on 10000 matches:{}\n".format(
            np.mean(scores.history['episode_reward'])))

        filename = input("Please enter a filename to save the trained weights [default_weights]: ").strip()
        save_model(filename or "default_weights", self.agent, self.env)


def train():
    env = None
    agent = None

    if input("Do you want to continue training a saved model? (y/n) [n]: ").strip().lower() == 'y':
        name = choose_model()
        if name:
            env, agent = load_model(name)
        else:
            print("Starting with a new model instead.")

    resumed = agent is not None
    if agent is None:
        env = blackjack_env(rules=_ask_rules())
        agent = _build_new(env)

    nb_steps = _ask_int("Number of training steps", 50000)
    lr = _ask_float("Learning rate", 1e-3)

    trainer = Trainer(env=env, agent=agent)
    trainer.train_agent(nb_steps=nb_steps, lr=lr, resume=resumed)
    return env, agent
