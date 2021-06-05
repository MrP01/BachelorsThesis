import pathlib

import matplotlib.pyplot as plt
import mnist
import numpy as np
import tensorflow as tf


def relu_taylor(x):
    """Taylor-approximated activation function
    ReLU(x) = 0.54738 + 0.59579 x + 0.090189 x^2 - 0.006137 x^3
    """
    polynomial = [0.006137, 0.090189, 0.59579, 0.54738]  # highest order first
    return tf.math.polyval(polynomial, x)


fig: plt.Figure = plt.figure()
axes: plt.Axes = fig.add_subplot(1, 1, 1)
x_ = tf.linspace(-10.0, 10.0, 100)
axes.plot(x_, tf.keras.activations.relu(x_), label=r"$y = \mathrm{relu}(x)$")
axes.plot(x_, relu_taylor(x_), label="$y = \mathrm{relu\_taylor}(x)$")
axes.set_xlabel(r"$x$")
axes.set_ylabel(r"$y$")
axes.legend()
fig.savefig(pathlib.Path.cwd().parent.parent / "thesis" / "figures" / "taylor-relu.png")

x_train, y_train, x_test, y_test = mnist.mnist()
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=relu_taylor),
    tf.keras.layers.Dense(10)
])
model.summary()
model.compile(optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)

freeze = pathlib.Path.cwd().parent / "classifier" / "data" / "models" / "simple"
w1, b1, w2, b2 = model.get_weights()
np.save(freeze / "w1.npy", w1)
np.save(freeze / "b1.npy", b1)
np.save(freeze / "w2.npy", w2)
np.save(freeze / "b2.npy", b2)
