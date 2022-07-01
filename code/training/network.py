"""The neural network used behind the scenes"""
import pathlib

import numpy as np
import tensorflow as tf


def relu_taylor(x):
    """Taylor-approximated activation function
    ReLU(x) = 0.54738 + 0.59579 x + 0.090189 x^2 - 0.006137 x^3
    """
    polynomial = [-0.006137, 0.090189, 0.59579, 0.54738]  # highest order first
    return tf.math.polyval(polynomial, x)


def train():
    """Design and Train the Neural Network using tensorflow, save the weights"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=relu_taylor),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Activation(tf.keras.activations.softmax),
        ]
    )
    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=1)
    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=30, validation_split=0.1, callbacks=[early_stopping])
    model.evaluate(x_test, y_test)

    freeze = pathlib.Path.cwd().parent / "classifier" / "data" / "models" / "simple"
    w1, b1, w2, b2 = model.get_weights()
    np.save(freeze / "w1.npy", w1)
    np.save(freeze / "b1.npy", b1)
    np.save(freeze / "w2.npy", w2)
    np.save(freeze / "b2.npy", b2)
    np.save(freeze / "training-history.npy", history)
    return w1, b1, w2, b2


if __name__ == "__main__":
    train()
