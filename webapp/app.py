import json
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # so the weights/ directory resolves at the repo root

# NOTE: keras-rl must never be imported in this process. Importing it calls
# tf.compat.v1.disable_eager_execution(), which puts Keras into v1 graph mode
# where loading a second model invalidates the first one's predict function
# ("Tensor ... was not found in the Graph"). Training (which needs keras-rl)
# runs in a spawned child process; advisors here use plain eager Keras.
import numpy as np
from flask import Flask, jsonify, render_template, request

from environments.blackjack_env import blackjack_env, RULES
from utils import model_store
from utils.qnet import build_q_network, DEFAULT_HIDDEN_LAYERS, DEFAULT_DROPOUT

app = Flask(__name__)
# always serve fresh JS/CSS; stale cached assets have caused UI bugs before
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

NAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,40}$')

# ---------------------------------------------------------------- training
#
# keras-rl2 runs tf.keras in v1 graph mode: training a new model inside this
# process invalidates the graph that already-loaded advisor models live in,
# after which their predict() raises "Tensor ... was not found in the Graph".
# Training therefore runs in a *spawned* child process and reports progress
# through an atomically-replaced JSON status file.

MP = multiprocessing.get_context('spawn')
TRAIN_LOCK = threading.Lock()
TRAIN = {'proc': None, 'dir': None, 'status_path': None, 'stop_path': None,
         'config': None, 'started_at': None, 'cache_cleared': True}

TERMINAL_PHASES = ('done', 'stopped', 'error')


def _child_train(cfg, status_path, stop_path):
    """Runs in a separate process; writes progress to status_path."""
    state = {'phase': 'starting', 'error': None, 'step': 0, 'episodes': 0,
             'rewards': [], 'stopped': False, 'eval': None, 'saved_as': None,
             'best': None, 'used_best': False}
    best_path = os.path.join(os.path.dirname(status_path), 'best.h5f')

    def flush():
        tmp = status_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, status_path)

    # keras-rl (graph mode) is safe to import here: this is a child process
    from agents.q_learning_agent import QLearningAgent
    from rl.callbacks import Callback
    from utils.evaluate import greedy_eval

    class _Metrics(Callback):
        def __init__(self, agent, rules, ckpt_every):
            self._last_flush = 0.0
            self._agent = agent
            self._rules = rules
            self._ckpt_every = ckpt_every

        def on_step_end(self, step, logs=None):
            state['step'] += 1
            if os.path.exists(stop_path):
                state['stopped'] = True
                raise KeyboardInterrupt  # keras-rl catches this and aborts cleanly
            if self._ckpt_every and state['step'] % self._ckpt_every == 0:
                mean, _ = greedy_eval(self._agent.model, self._rules,
                                      episodes=5000)
                if state['best'] is None or mean > state['best']['mean']:
                    self._agent.save(best_path)
                    state['best'] = {'step': state['step'],
                                     'mean': round(mean, 4)}
                flush()
            now = time.time()
            if now - self._last_flush >= 0.5:
                self._last_flush = now
                flush()

        def on_episode_end(self, episode, logs=None):
            logs = logs or {}
            state['episodes'] += 1
            state['rewards'].append(float(logs.get('episode_reward', 0.0)))

    flush()
    try:
        if cfg['base_model']:
            env, agent = model_store.load_model(cfg['base_model'])
        else:
            env = blackjack_env(rules=cfg['rules'])
            agent = QLearningAgent(env=env,
                                   total_states=env.observation_space.shape[0],
                                   actions=env.action_space.n,
                                   hidden_layers=cfg['layers'],
                                   dropout=cfg['dropout'])
        agent.compile(lr=cfg['lr'])

        # checkpoint the greedy policy every 5% of the run; short runs
        # finish before any checkpoint would be informative
        ckpt_every = cfg['steps'] // 20 if cfg['steps'] >= 5000 else 0
        metrics = _Metrics(agent, env.rules, ckpt_every)

        state['phase'] = 'training'
        flush()
        try:
            agent.train(nb_steps=cfg['steps'], callbacks=[metrics], verbose=0,
                        resume=bool(cfg['base_model']))
        finally:
            # keep the best checkpoint if the final weights score worse
            if state['best'] is not None:
                final_mean, _ = greedy_eval(agent.model, env.rules,
                                            episodes=5000)
                if state['best']['mean'] > final_mean:
                    agent.agent.load_weights(best_path)
                    state['used_best'] = True

        if cfg['eval_episodes'] > 0:
            state['phase'] = 'evaluating'
            flush()
            hist = agent.agent.test(env, nb_episodes=cfg['eval_episodes'],
                                    visualize=False, verbose=0)
            rewards = [float(r) for r in hist.history['episode_reward']]
            n = len(rewards)
            state['eval'] = {
                'episodes': n,
                'mean': round(float(np.mean(rewards)), 4),
                'std': round(float(np.std(rewards)), 4),
                'sem': round(float(np.std(rewards)) / np.sqrt(n), 4),
                'win_rate': round(sum(1 for r in rewards if r > 0) / n, 4),
                'loss_rate': round(sum(1 for r in rewards if r < 0) / n, 4),
                'push_rate': round(sum(1 for r in rewards if r == 0) / n, 4),
            }

        state['phase'] = 'saving'
        flush()
        model_store.save_model(cfg['name'], agent, env)
        state['saved_as'] = cfg['name']
        state['phase'] = 'stopped' if state['stopped'] else 'done'
    except Exception as exc:  # surface anything to the dashboard
        state['phase'] = 'error'
        state['error'] = str(exc)
    finally:
        flush()


def _read_child_status():
    path = TRAIN['status_path']
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return None


def _reward_series(rewards, window=200, max_points=300, min_window=25):
    """Rolling-mean reward, downsampled to at most max_points [episode, value]
    pairs. The warm-up region uses an expanding mean so the series starts
    near episode 0, but points averaging fewer than min_window episodes are
    skipped: a lucky first hand would otherwise paint a huge fake spike."""
    n = len(rewards)
    if n == 0:
        return []
    arr = np.asarray(rewards, dtype=float)
    w = int(min(window, max(1, n // 5)))
    cums = np.cumsum(arr)
    smooth = np.empty(n)
    head = min(w, n)
    smooth[:head] = cums[:head] / np.arange(1, head + 1)
    if n > w:
        smooth[w:] = (cums[w:] - cums[:-w]) / w
    start = min(n - 1, min_window - 1)
    idx = np.linspace(start, n - 1, min(max_points, n - start)).astype(int)
    return [[int(i), round(float(smooth[i]), 4)] for i in idx]


@app.route('/api/train/start', methods=['POST'])
def train_start():
    data = request.get_json(silent=True) or {}
    try:
        rules = data.get('rules', 'american')
        if rules not in RULES:
            raise ValueError('unknown rules: {}'.format(rules))
        layers = [int(x) for x in data.get('layers', DEFAULT_HIDDEN_LAYERS)]
        if not layers or len(layers) > 8 or any(not 1 <= u <= 2048 for u in layers):
            raise ValueError('layers must be 1-8 sizes between 1 and 2048')
        dropout = float(data.get('dropout', DEFAULT_DROPOUT))
        if not 0 <= dropout <= 0.9:
            raise ValueError('dropout must be between 0 and 0.9')
        steps = int(data.get('steps', 50000))
        if not 100 <= steps <= 1000000:
            raise ValueError('steps must be between 100 and 1000000')
        lr = float(data.get('lr', 1e-3))
        if not 0 < lr <= 1:
            raise ValueError('learning rate must be between 0 and 1')
        eval_episodes = int(data.get('eval_episodes', 1000))
        if not 0 <= eval_episodes <= 20000:
            raise ValueError('eval episodes must be between 0 and 20000')
        name = str(data.get('name', '')).strip() or 'default_weights'
        if not NAME_RE.match(name):
            raise ValueError('model name: letters, digits, _ and - only (max 40)')
        base_model = data.get('base_model') or None
        if base_model and base_model not in model_store.list_models():
            raise ValueError('unknown base model: {}'.format(base_model))
    except (TypeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400

    cfg = {'rules': rules, 'layers': layers, 'dropout': dropout, 'steps': steps,
           'lr': lr, 'eval_episodes': eval_episodes, 'name': name,
           'base_model': base_model}

    with TRAIN_LOCK:
        if TRAIN['proc'] is not None and TRAIN['proc'].is_alive():
            return jsonify({'error': 'a training run is already in progress'}), 409
        if TRAIN['dir']:
            shutil.rmtree(TRAIN['dir'], ignore_errors=True)
        run_dir = tempfile.mkdtemp(prefix='betql_train_')
        status_path = os.path.join(run_dir, 'status.json')
        stop_path = os.path.join(run_dir, 'stop')
        proc = MP.Process(target=_child_train, args=(cfg, status_path, stop_path),
                          daemon=True)
        proc.start()
        TRAIN.update(proc=proc, dir=run_dir, status_path=status_path,
                     stop_path=stop_path, config=cfg, started_at=time.time(),
                     cache_cleared=False)
    return jsonify({'ok': True})


@app.route('/api/train/status')
def train_status():
    proc = TRAIN['proc']
    if proc is None:
        return jsonify({'running': False, 'phase': 'idle', 'error': None,
                        'step': 0, 'total_steps': 0, 'episodes': 0,
                        'steps_per_sec': 0, 'elapsed': 0, 'recent_mean': None,
                        'series': [], 'live_rates': None, 'eval': None,
                        'saved_as': None, 'stopped': False, 'config': None,
                        'best': None, 'used_best': False})

    s = _read_child_status() or {'phase': 'starting', 'error': None, 'step': 0,
                                 'episodes': 0, 'rewards': [], 'stopped': False,
                                 'eval': None, 'saved_as': None, 'best': None,
                                 'used_best': False}
    alive = proc.is_alive()
    phase, error = s['phase'], s['error']
    if not alive and phase not in TERMINAL_PHASES:
        phase, error = 'error', 'training process exited unexpectedly'
    if not alive and not TRAIN['cache_cleared']:
        with AGENT_LOCK:
            AGENT_CACHE.pop(TRAIN['config']['name'], None)  # drop stale advisor weights
        TRAIN['cache_cleared'] = True

    rewards = s['rewards']
    recent = rewards[-200:]
    live_rates = None
    if rewards:
        rr = rewards[-500:]
        n = len(rr)
        live_rates = {'episodes': n,
                      'win_rate': round(sum(1 for r in rr if r > 0) / n, 4),
                      'push_rate': round(sum(1 for r in rr if r == 0) / n, 4),
                      'loss_rate': round(sum(1 for r in rr if r < 0) / n, 4)}
    total_steps = TRAIN['config']['steps']
    elapsed = time.time() - TRAIN['started_at']
    return jsonify({
        'running': alive, 'phase': phase, 'error': error,
        'step': s['step'], 'total_steps': total_steps,
        'episodes': s['episodes'],
        'steps_per_sec': round(s['step'] / elapsed, 1) if elapsed > 0 and s['step'] else 0,
        'elapsed': round(elapsed, 1),
        'recent_mean': round(float(np.mean(recent)), 4) if recent else None,
        'series': _reward_series(rewards),
        'live_rates': live_rates,
        'eval': s['eval'], 'saved_as': s['saved_as'],
        'stopped': s['stopped'], 'config': TRAIN['config'],
        'best': s.get('best'), 'used_best': s.get('used_best', False),
    })


@app.route('/api/train/stop', methods=['POST'])
def train_stop():
    proc = TRAIN['proc']
    if proc is None or not proc.is_alive():
        return jsonify({'error': 'no training run in progress'}), 409
    with open(TRAIN['stop_path'], 'w'):
        pass
    return jsonify({'ok': True})


# ---------------------------------------------------------------- models

@app.route('/api/models')
def models():
    out = []
    for name in model_store.list_models():
        try:
            with open(os.path.join(model_store.WEIGHTS_DIR, name + '.json')) as f:
                meta = json.load(f)
        except (IOError, ValueError):
            meta = {}
        out.append({'name': name,
                    'rules': meta.get('rules', 'american'),
                    'hidden_layers': meta.get('hidden_layers', []),
                    'dropout': meta.get('dropout')})
    return jsonify(out)


# ---------------------------------------------------------------- play

GAMES = {}
GAMES_MAX = 50
AGENT_CACHE = {}
AGENT_LOCK = threading.RLock()


class Advisor:
    """Inference-only Q-network. DQNAgent.save_weights stores exactly the
    inner Keras model's weights, so plain Keras can serve suggestions."""

    def __init__(self, name):
        meta = model_store.load_metadata(name)
        self.model = build_q_network(meta['total_states'], meta['actions'],
                                     meta['hidden_layers'],
                                     meta.get('dropout', DEFAULT_DROPOUT),
                                     dueling=meta.get('dueling', False))
        status = self.model.load_weights(
            os.path.join(model_store.WEIGHTS_DIR, name + '.h5f'))
        if status is not None:
            status.expect_partial()

    def suggest(self, observation):
        x = np.asarray(observation, dtype=np.float32).reshape(1, 1, -1)
        return int(np.argmax(self.model.predict(x)[0]))


def _get_agent(name):
    with AGENT_LOCK:
        if name not in AGENT_CACHE:
            AGENT_CACHE[name] = Advisor(name)
        return AGENT_CACHE[name]


def _hand_outcomes(env):
    """Per-hand result labels, mirroring blackjack_env._resolve."""
    dealer_score = int(env.calculate_score(env.dealer_hand))
    dealer_natural = len(env.dealer_hand) == 2 and dealer_score == 21
    outcomes = []
    for hand, doubled in zip(env.hands, env.doubled):
        score = int(env.calculate_score(hand))
        natural = (not env.split_done and not doubled
                   and len(hand) == 2 and score == 21)
        if score > 21:
            outcomes.append('bust')
        elif dealer_natural and not natural:
            outcomes.append('lose')
        elif natural and dealer_natural:
            outcomes.append('push')
        elif natural:
            outcomes.append('blackjack')
        elif dealer_score > 21 or score > dealer_score:
            outcomes.append('win')
        elif score < dealer_score:
            outcomes.append('lose')
        else:
            outcomes.append('push')
    return outcomes


def _game_state(game, reward=None, done=False):
    env = game['env']
    valid = [] if done else env.valid_actions()
    suggestion = None
    if not done and game['agent_name']:
        try:
            obs = env.get_state(env._get_obs())
            with AGENT_LOCK:
                suggestion = _get_agent(game['agent_name']).suggest(obs)
            if suggestion not in valid:
                suggestion = 1  # the env falls back to hit for invalid choices
        except Exception as exc:
            # an unusable advisor should never block the game itself
            app.logger.warning('advisor %s failed: %s', game['agent_name'], exc)
            suggestion = None
    state = {
        'game_id': game['id'],
        'rules': env.rules,
        'hands': [[int(c) for c in hand] for hand in env.hands],
        'scores': [int(env.calculate_score(h)) for h in env.hands],
        'doubled': list(env.doubled),
        'current': int(env.current),
        'dealer': [int(c) for c in (env.dealer_hand if done else env.dealer_hand[:1])],
        'dealer_hidden': (not done) and env.rules == 'american',
        'dealer_score': int(env.calculate_score(env.dealer_hand)) if done else None,
        'valid_actions': valid,
        'suggestion': suggestion,
        'done': done,
        'reward': float(reward) if reward is not None else None,
        'outcomes': _hand_outcomes(env) if done else None,
        'advisor': game['agent_name'],
    }
    return state


@app.route('/api/game/new', methods=['POST'])
def game_new():
    data = request.get_json(silent=True) or {}
    rules = data.get('rules', 'american')
    if rules not in RULES:
        return jsonify({'error': 'unknown rules: {}'.format(rules)}), 400
    agent_name = data.get('model') or None
    if agent_name and agent_name not in model_store.list_models():
        return jsonify({'error': 'unknown model: {}'.format(agent_name)}), 400

    if len(GAMES) >= GAMES_MAX:
        oldest = min(GAMES.values(), key=lambda g: g['created'])
        GAMES.pop(oldest['id'], None)

    env = blackjack_env(rules=rules)
    env.reset()
    game = {'id': uuid.uuid4().hex, 'env': env, 'agent_name': agent_name,
            'done': False, 'created': time.time(), 'lock': threading.Lock()}
    GAMES[game['id']] = game
    return jsonify(_game_state(game))


@app.route('/api/game/<game_id>/action', methods=['POST'])
def game_action(game_id):
    game = GAMES.get(game_id)
    if game is None:
        return jsonify({'error': 'unknown game'}), 404
    data = request.get_json(silent=True) or {}
    try:
        action = int(data.get('action'))
    except (TypeError, ValueError):
        return jsonify({'error': 'action must be an integer'}), 400

    # serialize concurrent requests (double clicks) so the env never
    # steps twice from the same state
    with game['lock']:
        if game['done']:
            return jsonify({'error': 'game is already over'}), 409
        if action not in game['env'].valid_actions():
            return jsonify({'error': 'invalid action for this state'}), 400
        _, reward, done, _ = game['env'].step(action)
        game['done'] = bool(done)
        return jsonify(_game_state(game, reward=reward, done=bool(done)))


# ---------------------------------------------------------------- pages

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('BETQL_PORT', sys.argv[1] if len(sys.argv) > 1 else 5050))
    print('BetQL web UI running at http://127.0.0.1:{}'.format(port))
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
