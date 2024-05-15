import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
tf.get_logger().setLevel('ERROR')
class QLearningAgent:
    def __init__(self, env, total_states, actions):
        self.env = env
        self.total_states = total_states
        self.actions = actions
        self.model = self.build_model()
        self.agent = self.build_agent()
        tf.get_logger().setLevel('ERROR')

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.total_states)))
        #model.add(Dense(512, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(256, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(self.actions, activation='linear'))
    
        return model

    def build_agent(self):
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=self.model, memory=memory, policy=policy,
                       nb_actions=self.actions, nb_steps_warmup=20, target_model_update=1e-2)
        return dqn

    def compile(self, lr=1e-3):
        self.agent.compile(Adam(lr=lr), metrics=['mse'])

    def train(self, nb_steps=50000):
        self.agent.fit(self.env, nb_steps=nb_steps, visualize=False, verbose=1)

    def save(self, filepath):
        self.agent.save_weights(filepath, overwrite=True)

    def load(self, filepath):
        self.agent.load_weights(filepath)
