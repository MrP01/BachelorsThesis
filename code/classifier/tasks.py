import pathlib

import invoke
import mnist
import numpy as np


@invoke.task()
def fetch_training_data(ctx, target="data/mnist"):
    """Download MNIST training data"""
    x_train, y_train, x_test, y_test = mnist.mnist()
    target = pathlib.Path(target).resolve()
    ctx.run(f"mkdir -p {target}")
    np.save(target / "x-train.npy", x_train)
    np.save(target / "y-train.npy", y_train)
    np.save(target / "x-test.npy", x_test)
    np.save(target / "y-test.npy", y_test)
