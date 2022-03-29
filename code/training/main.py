import pathlib

import matplotlib.pyplot as plt
import network
import tensorflow as tf


THESIS = pathlib.Path.cwd().parent.parent / "thesis"


def plot_metric(history, metric):
    """Plots the training development"""
    train_metrics = history.history[metric]
    val_metrics = history.history["val_" + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title(f"Training and validation {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([f"train_{metric}", f"val_{metric}"])
    plt.show()


def plot_relu_taylor():
    """Plots the approximated RELU function"""
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    x_ = tf.linspace(-5.0, 10.0, 100)
    axes.plot(x_, tf.keras.activations.relu(x_), label=r"$y = \mathrm{relu}(x)$")
    axes.plot(x_, network.relu_taylor(x_), label=r"$y = \mathrm{relu\_taylor}(x)$")
    axes.set_xlabel(r"$x$")
    axes.set_ylabel(r"$y$")
    axes.legend()
    fig.savefig(THESIS / "figures" / "taylor-relu.png")


def main():
    plot_relu_taylor()
    network.train()


if __name__ == "__main__":
    main()
