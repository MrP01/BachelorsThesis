import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def relu_taylor(x):
    """Taylor-approximated activation function
    ReLU(x) = 0.54738 + 0.59579 x + 0.090189 x^2 - 0.006137 x^3
    """
    polynomial = [-0.006137, 0.090189, 0.59579, 0.54738]  # highest order first
    return tf.math.polyval(polynomial, x)


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history["val_" + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title("Training and validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, "val_" + metric])
    plt.show()


def plot_relu_taylor():
    fig: plt.Figure = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    x_ = tf.linspace(-5.0, 10.0, 100)
    axes.plot(x_, tf.keras.activations.relu(x_), label=r"$y = \mathrm{relu}(x)$")
    axes.plot(x_, relu_taylor(x_), label=r"$y = \mathrm{relu\_taylor}(x)$")
    axes.set_xlabel(r"$x$")
    axes.set_ylabel(r"$y$")
    axes.legend()
    fig.savefig(pathlib.Path.cwd().parent.parent / "thesis" / "figures" / "taylor-relu.png")


def main():
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

    # TODO: create and report confusion matrix

    freeze = pathlib.Path.cwd().parent / "classifier" / "data" / "models" / "simple"
    w1, b1, w2, b2 = model.get_weights()
    np.save(freeze / "w1.npy", w1)
    np.save(freeze / "b1.npy", b1)
    np.save(freeze / "w2.npy", w2)
    np.save(freeze / "b2.npy", b2)


if __name__ == "__main__":
    main()
