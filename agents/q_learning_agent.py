import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from utils.qnet import build_q_network, DEFAULT_HIDDEN_LAYERS, DEFAULT_DROPOUT

tf.get_logger().setLevel('ERROR')


class QLearningAgent:
    def __init__(self, env, total_states, actions,
                 hidden_layers=None, dropout=DEFAULT_DROPOUT, dueling=True):
        self.env = env
        self.total_states = total_states
        self.actions = actions
        self.hidden_layers = list(hidden_layers) if hidden_layers else list(DEFAULT_HIDDEN_LAYERS)
        self.dropout = dropout
        self.dueling = dueling
        self.compiled = False
        self.model = self.build_model()
        self.agent = self.build_agent()

    def build_model(self):
        return build_q_network(self.total_states, self.actions,
                               self.hidden_layers, self.dropout,
                               dueling=self.dueling)

    def build_agent(self):
        # epsilon-greedy annealed over the run (nb_steps is rescaled by
        # train()); exploring at a constant rate forever keeps the policy
        # from refining late in training
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                      value_max=1.0, value_min=0.05,
                                      value_test=0.0, nb_steps=40000)
        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=self.model, memory=memory, policy=policy,
                       nb_actions=self.actions, nb_steps_warmup=20,
                       target_model_update=1e-2)
        return dqn

    def compile(self, lr=1e-3):
        self.agent.compile(Adam(lr=lr), metrics=['mse'])
        self.compiled = True

    def train(self, nb_steps=50000, callbacks=None, verbose=1, resume=False):
        if not self.compiled:
            self.compile()
        # finish annealing at 80% of the run so the last 20% refines
        # near-greedily; fit() resets the step counter, so continued
        # training re-anneals over its own run. A resumed model already
        # has a competent policy: restarting exploration at eps=1.0 would
        # fill the replay buffer with random play and degrade it.
        self.agent.policy.value_max = 0.2 if resume else 1.0
        self.agent.policy.nb_steps = max(1, int(nb_steps * 0.8))
        self.agent.nb_steps_warmup = min(1000, max(20, nb_steps // 10))
        self.agent.fit(self.env, nb_steps=nb_steps, visualize=False,
                       verbose=verbose, callbacks=callbacks)

    def suggest(self, observation):
        """Greedy action for the given observation (no exploration)."""
        was_training = self.agent.training
        self.agent.training = False
        action = self.agent.forward(observation)
        self.agent.training = was_training
        return int(action)

    def save(self, filepath):
        self.agent.save_weights(filepath, overwrite=True)

    def load(self, filepath):
        # keras-rl builds the target model in compile(); weights cannot
        # be restored before that happens.
        if not self.compiled:
            self.compile()
        self.agent.load_weights(filepath)
