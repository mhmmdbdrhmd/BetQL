import numpy as np

from environments.blackjack_env import blackjack_env


def greedy_eval(model, rules, episodes=3000, batch=512):
    """Mean greedy-policy reward over `episodes`, batching many games per
    predict call so mid-training checkpoints stay cheap. Works on the inner
    Keras Q-network directly and never touches keras-rl agent state."""
    rewards = []
    while len(rewards) < episodes:
        n = min(batch, episodes - len(rewards))
        envs = [blackjack_env(rules=rules) for _ in range(n)]
        states = [e.reset() for e in envs]
        totals = [0.0] * n
        active = list(range(n))
        while active:
            x = np.asarray([states[i] for i in active],
                           dtype=np.float32).reshape(len(active), 1, -1)
            actions = np.argmax(model.predict_on_batch(x), axis=1)
            still = []
            for k, i in enumerate(active):
                state, reward, done, _ = envs[i].step(int(actions[k]))
                totals[i] += reward
                states[i] = state
                if not done:
                    still.append(i)
            active = still
        rewards.extend(totals)
    return float(np.mean(rewards)), rewards
