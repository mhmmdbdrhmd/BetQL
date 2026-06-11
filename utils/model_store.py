import json
import os

WEIGHTS_DIR = 'weights'


def _weights_path(name):
    return os.path.join(WEIGHTS_DIR, name + '.h5f')


def _meta_path(name):
    return os.path.join(WEIGHTS_DIR, name + '.json')


def list_models():
    """Names of saved models (weights checkpoint + metadata present)."""
    if not os.path.isdir(WEIGHTS_DIR):
        return []
    names = []
    for fname in sorted(os.listdir(WEIGHTS_DIR)):
        if fname.endswith('.json'):
            name = fname[:-len('.json')]
            if os.path.exists(_weights_path(name) + '.index'):
                names.append(name)
    return names


def save_model(name, agent, env):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    agent.save(_weights_path(name))
    metadata = {
        'rules': env.rules,
        'total_states': agent.total_states,
        'actions': agent.actions,
        'hidden_layers': agent.hidden_layers,
        'dropout': agent.dropout,
        'dueling': agent.dueling,
    }
    with open(_meta_path(name), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Saved model '{}' to {}/".format(name, WEIGHTS_DIR))


def load_metadata(name):
    with open(_meta_path(name)) as f:
        return json.load(f)


def load_model(name):
    """Rebuild env + agent from saved metadata and restore weights."""
    # imported lazily: pulling in keras-rl disables TF eager execution
    # process-wide, which inference-only users must avoid
    from agents.q_learning_agent import QLearningAgent
    from environments.blackjack_env import blackjack_env

    metadata = load_metadata(name)
    env = blackjack_env(rules=metadata.get('rules', 'american'))
    agent = QLearningAgent(env=env,
                           total_states=metadata['total_states'],
                           actions=metadata['actions'],
                           hidden_layers=metadata['hidden_layers'],
                           dropout=metadata.get('dropout', 0.5),
                           dueling=metadata.get('dueling', False))
    agent.load(_weights_path(name))
    print("Loaded model '{}' (rules: {}, layers: {})".format(
        name, env.rules, agent.hidden_layers))
    return env, agent


def choose_model():
    """Interactively pick a saved model. Returns a name or None."""
    models = list_models()
    if not models:
        print("No saved models found in '{}/'.".format(WEIGHTS_DIR))
        return None
    print("Available models:")
    for i, name in enumerate(models, 1):
        print("  {}. {}".format(i, name))
    choice = input("Enter a model name or number: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    if choice in models:
        return choice
    print("'{}' is not a saved model.".format(choice))
    return None
