import glob
import pathlib

import invoke
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

THESIS = pathlib.Path(__file__).resolve().parent.parent.parent / "thesis"
# plt.style.use("ggplot")


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
    plt.show()


@invoke.task()
def plot_relu_taylor(ctx):
    """Plots the approximated RELU function"""
    from training.network import relu_taylor, tf

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
    # fig.savefig(THESIS / "figures" / "ciphertext-visualisation.png", bbox_inches="tight")
    # tikzplotlib.clean_figure(fig)
    # tikzplotlib.save(
    #     THESIS / "figures" / "generated" / "ciphertext-visualisation.tex",
    #     figure=fig,
    #     axis_width=r"0.2\linewidth",
    #     axis_height=r"0.2\linewidth",
    # )
    plt.show()
