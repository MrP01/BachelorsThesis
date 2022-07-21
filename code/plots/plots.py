"""Invoke tasks for plotting"""
# pylint: disable=unused-argument
import glob
import pathlib

import invoke
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

CODE = pathlib.Path(__file__).resolve().parent.parent
THESIS = CODE.parent / "thesis"
MODEL = CODE / "classifier" / "data" / "models" / "simple"
MNIST = CODE / "classifier" / "data" / "mnist"


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
        override_externals=True,
        tex_relative_path_to_data="figures/generated/",
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
        override_externals=True,
        tex_relative_path_to_data="figures/generated/",
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
    print(f"Found {len(pairs)} pairs.")

    fig = plt.figure()
    for i, (ciphertext, plain) in enumerate(pairs, start=1):
        axes: plt.Axes = fig.add_subplot(2, len(pairs), i)
        axes.imshow(plain)
        axes.set_axis_off()
    for i, (ciphertext, plain) in enumerate(pairs, start=1):
        axes: plt.Axes = fig.add_subplot(2, len(pairs), len(pairs) + i)
        axes.imshow(ciphertext)
        axes.set_axis_off()


def plot_confusion_matrix(predictions, y_test):
    """Creates and plots the confusion matrix"""
    import tensorflow as tf

    matrix = np.array(tf.math.confusion_matrix(y_test, predictions))
    print("Confusion Matrix:", matrix)
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    im = axes.matshow(np.log2(matrix + 1))
    threshold = im.norm(matrix.max()) / 2.0
    cbar = axes.figure.colorbar(im, ax=axes)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "black" if im.norm(matrix[i, j]) > threshold else "white"
            im.axes.text(j, i, matrix[i, j], horizontalalignment="center", verticalalignment="center", color=color)
    axes.set_xlabel("True Digit")
    axes.set_ylabel("Classification")
    fig.savefig(THESIS / "figures" / "confusion-matrix.png")
    tikzplotlib.save(
        THESIS / "figures" / "generated" / "confusion-matrix.tex",
        override_externals=True,
        # tex_relative_path_to_data="figures/generated/",
    )


def plot_misclassifications(x_test, predictions, y_test):
    """Given the predictions and actual labels, plot some images that were misclassified"""
    from PIL import Image

    misclassifications = predictions != y_test
    for number in range(10):
        i = list(predictions[misclassifications]).index(number)  # finds first image predicted as number
        x = x_test[misclassifications][i]
        img = Image.fromarray(x * 255).convert("P")
        img.save(THESIS / "figures" / "generated" / f"mnist-misclassification-{number}.png")


def precision_and_recall(predictions: np.ndarray, y_test: np.ndarray):
    """For the given predictions and true labels, analyse"""
    precisions = []
    recalls = []
    for digit in range(10):
        true_value_is = y_test == digit
        prediction_is = predictions == digit
        true_positives = sum(prediction_is[true_value_is])
        false_positives = sum(prediction_is[~true_value_is])
        true_negatives = sum(~prediction_is[~true_value_is])
        false_negatives = sum(~prediction_is[true_value_is])
        assert true_positives + false_positives == sum(prediction_is)
        assert true_negatives + false_negatives == sum(~prediction_is)
        assert true_positives + true_negatives == sum(prediction_is == true_value_is)
        precisions.append(true_positives / (true_positives + false_positives))
        recalls.append(true_positives / (true_positives + false_negatives))
    print("Precision:", precisions)
    print("Recall:", recalls)
    print("Average Precision:", np.mean(precisions))
    print("Average Recall:", np.mean(recalls))
    return precisions, recalls


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
        override_externals=True,
        tex_relative_path_to_data="figures/generated/",
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
        tikzplotlib.save(
            THESIS / "figures" / "generated" / f"{name}.tex",
            override_externals=True,
            tex_relative_path_to_data="figures/generated/",
        )


@invoke.task()
def export_mnist_images(ctx):
    """Export some MNIST images"""
    from PIL import Image

    x_test = np.load(MNIST / "x-test.npy")
    y_test = list(np.load(MNIST / "y-test.npy"))
    for number in range(10):
        i = y_test.index(number)  # finds first image of this value
        img = Image.fromarray(x_test[i] * 255).convert("P")
        img.save(THESIS / "figures" / "generated" / f"mnist-test-{number}.png")
