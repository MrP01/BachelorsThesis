#!/usr/bin/env python3
"""Interaction with the neural network"""
import pathlib

import matplotlib.pyplot as plt
import network

THESIS = pathlib.Path(__file__).resolve().parent.parent.parent / "thesis"


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


def plot_weights(w, b, filename):
    """Matshow of the weights and biases"""
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=(5, 1))
    axes: plt.Axes = fig.add_subplot(gs[0])
    axes.imshow(w, aspect="auto")
    axes.set_title("Weights")
    axes: plt.Axes = fig.add_subplot(gs[1])
    axes.imshow(b.reshape((1, b.shape[0])), aspect="auto")
    axes.set_title("Biases")
    fig.savefig(THESIS / "figures" / filename)


def main():
    """Train the network and plot the layer weights"""
    w1, b1, w2, b2 = network.train()
    plot_weights(w1, b1, "layer-1.png")
    plot_weights(w2, b2, "layer-2.png")
    plt.show()


if __name__ == "__main__":
    main()
