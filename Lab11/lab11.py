import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

from tensorflow import keras
from keras import optimizers

def build_model(n_hidden, n_neurons, optimizer, learning_rate, momentum=0):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    for i in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation = "relu"))
    if optimizer == 'sgd':
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    elif optimizer == 'nesterov':
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True), loss="mse", metrics=["mae"])
    elif optimizer == 'momentum':
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), loss="mse", metrics=["mae"])
    elif optimizer == 'adam':
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model

es = tf.keras.callbacks.EarlyStopping(patience=10,
                                      min_delta=1.0,
                                      verbose=1)

import os
root_logdir = os.path.join(os.curdir, "tb_logs")
def get_run_logdir(name, value):
    import time
    ts = int(time.time())
    run_id = "{}_{}_{}".format(ts, name, value)
    return os.path.join(root_logdir, run_id)

reslr = []
for lr in [10**(-4), 10**(-5), 10**(-6)]:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir("lr", lr))
    model = build_model(1, 25, 'sgd', lr)
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es])
    score = model.evaluate(X_test, y_test)
    print(score[0], score[1])
    reslr.append((lr, score[0], score[1]))
print(reslr)

reshl = []
for hl in [0, 1, 2, 3]:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir("hl", hl))
    model = build_model(hl, 25, 'sgd', 10**(-5))
    #history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es])
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es])
    score = model.evaluate(X_test, y_test)
    print(score[0], score[1])
    #reshl.append((hl, history.history['loss'][-1], history.history['mae'][-1]))
    reshl.append((hl, score[0], score[1]))
print(reshl)

resnn = []
for nn in [5, 25, 125]:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir("nn", nn))
    model = build_model(1, nn, 'sgd', 10**(-5))
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es])
    score = model.evaluate(X_test, y_test)
    resnn.append((nn, score[0], score[1]))
print(resnn)

resopt = []
for opt in ['sgd', 'nesterov', 'momentum', 'adam']:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir("opt", opt))
    model = build_model(1, 25, opt, 10**(-5))
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es])
    score = model.evaluate(X_test, y_test)
    resopt.append((opt, score[0], score[1]))
print(resopt)

resmom = []
for mom in [0.1, 0.5, 0.9]:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir("mom", mom))
    model = build_model(1, 25, 'momentum', 10**(-5), mom)
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es])
    score = model.evaluate(X_test, y_test)
    resmom.append((mom, score[0], score[1]))
print(resmom)

param_distribs = {
"model__n_hidden": [0, 1, 2, 3],
"model__n_neurons": [5, 25, 125],
"model__learning_rate": [10**(-4), 10**(-5), 10**(-6)],
"model__optimizer": ['sgd', 'nesterov', 'momentum', 'adam'],
"model__momentum": [0.1, 0.5, 0.9]
}

