from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Lambda
from tensorflow.keras.models import Model, Sequential

DEFAULT_HIDDEN_LAYERS = [128, 64, 32]
DEFAULT_DROPOUT = 0.0


def _dueling_combine(a):
    # Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
    return K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1,
                                                          keepdims=True)


def build_q_network(total_states, actions, hidden_layers, dropout,
                    dueling=False):
    """Q-network shared by training (keras-rl) and serving (plain Keras).

    The dueling head lives here rather than in keras-rl's
    enable_dueling_network so that saved checkpoints can always be
    rebuilt and loaded without importing keras-rl.
    """
    if not dueling:
        model = Sequential()
        model.add(Flatten(input_shape=(1, total_states)))
        for units in hidden_layers:
            model.add(Dense(units, activation='relu'))
            if dropout > 0:
                model.add(Dropout(dropout))
        model.add(Dense(actions, activation='linear'))
        return model

    inp = Input(shape=(1, total_states))
    x = Flatten()(inp)
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
    x = Dense(actions + 1, activation='linear')(x)
    out = Lambda(_dueling_combine, output_shape=(actions,))(x)
    return Model(inputs=inp, outputs=out)
