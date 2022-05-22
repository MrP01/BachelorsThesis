import pathlib

import invoke
import matplotlib.pyplot as plt
import numpy as np

THESIS = pathlib.Path(__file__).resolve().parent.parent.parent / "thesis"


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
    plt.show()
