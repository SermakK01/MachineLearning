from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)

import pandas as pd
from sklearn.model_selection import train_test_split
X = iris.data[["petal length (cm)", "petal width (cm)"]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train_0 = (y_train == 0).astype(int)
y_test_0 = (y_test == 0).astype(int)
y_train_1 = (y_train == 1).astype(int)
y_test_1 = (y_test == 1).astype(int)
y_train_2 = (y_train == 2).astype(int)
y_test_2 = (y_test == 2).astype(int)

from sklearn.linear_model import Perceptron
per_clf_0 = Perceptron()
per_clf_1 = Perceptron()
per_clf_2 = Perceptron()
per_clf_0.fit(X_train, y_train_0)
per_clf_1.fit(X_train, y_train_1)
per_clf_2.fit(X_train, y_train_2)

y_pred_0_train = per_clf_0.predict(X_train)
y_pred_1_train = per_clf_1.predict(X_train)
y_pred_2_train = per_clf_2.predict(X_train)
y_pred_0_test = per_clf_0.predict(X_test)
y_pred_1_test = per_clf_1.predict(X_test)
y_pred_2_test = per_clf_2.predict(X_test)

from sklearn.metrics import accuracy_score

x1 = accuracy_score(y_train_0, y_pred_0_train)
x2 = accuracy_score(y_train_1, y_pred_1_train)
x3 = accuracy_score(y_train_2, y_pred_2_train)
x4 = accuracy_score(y_test_0, y_pred_0_test)
x5 = accuracy_score(y_test_1, y_pred_1_test)
x6 = accuracy_score(y_test_2, y_pred_2_test)

list1 = [(x1,x4),(x2,x5),(x3,x6)]
print(list1)

import pickle

with open('per_acc.pkl', 'wb') as f:
    pickle.dump(list1,f)

list2 = []

for x in [per_clf_0, per_clf_1, per_clf_2]:
    w_0 = x.intercept_[0]
    w_1 = x.coef_[0, 0]
    w_2 = x.coef_[0, 1]
    list2.append((w_0, w_1, w_2))
_
with open('per_wght.pkl', 'wb') as f:
    pickle.dump(list2,f)

print(list2)

import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
model.add(keras.layers.Dense(1, activation="sigmoid", use_bias=True))

model.summary()

model.compile(loss="binary_crossentropy", optimizer="sgd")

history = model.fit(X, y, epochs=100, verbose=False)
print(history.history['loss'])

model.predict(X)

model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
model2.add(keras.layers.Dense(1, activation="sigmoid"))
model2.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))
history = model2.fit(X, y, epochs=100, verbose=False)

model2.predict(X)

model3 = keras.models.Sequential()
model3.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
model3.add(keras.layers.Dense(1, activation="sigmoid"))
model3.compile(loss='mean_squared_error',
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), metrics=['binary_accuracy'])
history = model3.fit(X, y, epochs=100, verbose=False)

model3.predict(X)

found = False
while not found:
    model4 = keras.models.Sequential()
    model4.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
    model4.add(keras.layers.Dense(1, activation="sigmoid", use_bias=True))
    model4.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=["accuracy"])
    history = model4.fit(X, y, epochs=100, verbose=False)
    results = model4.predict(X)
    if results[0]<0.1 and results[1]>0.9 and results[2]>0.9 and results[0]<0.1:
        found = True

results

weights=model4.get_weights()

with open('mlp_xor_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)
