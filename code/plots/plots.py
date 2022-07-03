"""Invoke tasks for plotting"""
# pylint: disable=unused-argument
import glob
import pathlib

import invoke
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

THESIS = pathlib.Path(__file__).resolve().parent.parent.parent / "thesis"
MODEL = pathlib.Path(__file__).resolve().parent.parent / "classifier" / "data" / "models" / "simple"


@invoke.task()
def plot_rotation_error(ctx, layer1_csv="plots/rotdiff-layer1.csv", layer2_csv="plots/rotdiff-layer2.csv"):
    """Plots the increasing rotation (n) error during the diagonal method multiplications"""
    fig = plt.figure()
    for i, layer in enumerate((np.loadtxt(layer1_csv), np.loadtxt(layer2_csv)), start=1):
        axes: plt.Axes = fig.add_subplot(2, 1, i)
        axes.semilogy(layer, label=r"$\sum (x - x_{true})^2$")
        # axes.plot(np.sqrt(layer), label=r"$\sqrt{\sum (x - x_{true})^2}$")
        axes.set_xlabel(r"$n$")
        axes.set_ylabel(r"error")
        axes.legend()
        axes.set_title(f"Layer {i}")
    fig.savefig(THESIS / "figures" / "rotation-error.png")
    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(
        THESIS / "figures" / "rotation-error.tex",
        figure=fig,
        axis_width=r"0.7\linewidth",
        axis_height=r"0.25\linewidth",
    )


@invoke.task()
def plot_relu_taylor(ctx):
    """Plots the approximated RELU function"""
    from network import relu_taylor, tf

    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    x_ = tf.linspace(-5.0, 10.0, 100)
    axes.plot(x_, tf.keras.activations.relu(x_), label=r"$y = \mathrm{relu}(x)$")
    axes.plot(x_, relu_taylor(x_), label=r"$y = \mathrm{relu\_taylor}(x)$")
    axes.set_xlabel(r"$x$")
    axes.set_ylabel(r"$y$")
    axes.legend()
    fig.savefig(THESIS / "figures" / "taylor-relu.png")
    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(
        THESIS / "figures" / "generated" / "taylor-relu.tex",
        figure=fig,
        axis_width=r"0.7\linewidth",
        axis_height=r"0.4\linewidth",
    )


@invoke.task()
def plot_ciphertext(ctx):
    """Plots a pixel representation of the ciphertext"""
    pairs = []
    directory = pathlib.Path("classifier/data/ciphertext-visualisation/")
    for file in glob.glob("*-ciphertext.npy", root_dir=str(directory)):
        ciphertext = np.load(directory / file)
        plain = np.load(directory / f"{file.split('-')[0]}-plain.npy").reshape((28, 28))
        pairs.append((ciphertext, plain))

    fig = plt.figure()
    for i, (ciphertext, plain) in enumerate(pairs, start=1):
        axes: plt.Axes = fig.add_subplot(2, len(pairs), i)
        axes.imshow(plain)
        axes.set_axis_off()
    for i, (ciphertext, plain) in enumerate(pairs, start=1):
        axes: plt.Axes = fig.add_subplot(2, len(pairs), len(pairs) + i)
        axes.imshow(ciphertext)
        axes.set_axis_off()


def confusion_matrix(model, x_test, y_test):
    """Creates and plots the confusion matrix"""
    matrix = np.zeros((10, 10))
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    axes.matshow(matrix)
    axes.set_xlabel("True Digit")
    axes.set_ylabel("Classification")
    fig.savefig(THESIS / "figures" / "confusion-matrix.png")
    tikzplotlib.save(THESIS / "figures" / "generated" / "confusion-matrix.tex")


def plot_metric(history):
    """Plots the training history (error, accuracy development)"""
    fig = plt.figure()
    for i, metric in enumerate(("accuracy", "loss"), start=1):
        train_metrics = history.history[metric]
        val_metrics = history.history["val_" + metric]
        epochs = range(1, len(train_metrics) + 1)
        axes: plt.Axes = fig.add_subplot(2, 1, i)
        axes.plot(epochs, train_metrics, label=f"training {metric}")
        axes.plot(epochs, val_metrics, label=f"validation {metric}")
        axes.set_xlabel("Epochs")
        axes.set_ylabel(metric)
        axes.legend()
    fig.savefig(THESIS / "figures" / "training-history.png")
    tikzplotlib.save(
        THESIS / "figures" / "generated" / "training-history.tex",
        axis_width=r"0.7\linewidth",
        axis_height=r"0.25\linewidth",
    )


@invoke.task()
def plot_weights(ctx):
    """Matshow of the weights and biases"""
    for name, (w, b) in [
        ("layer-1", (np.load(MODEL / "w1.npy"), np.load(MODEL / "b1.npy"))),
        ("layer-2", (np.load(MODEL / "w2.npy"), np.load(MODEL / "b2.npy"))),
    ]:
        fig = plt.figure()
        gs = fig.add_gridspec(2, 1, height_ratios=(5, 1))
        axes: plt.Axes = fig.add_subplot(gs[0])
        axes.imshow(w, aspect="auto")
        axes.set_title("Weights")
        axes: plt.Axes = fig.add_subplot(gs[1])
        axes.imshow(b.reshape((1, b.shape[0])), aspect="auto")
        axes.set_title("Biases")
        fig.savefig(THESIS / "figures" / f"{name}.png")
        tikzplotlib.save(THESIS / "figures" / "generated" / f"{name}.tex")
