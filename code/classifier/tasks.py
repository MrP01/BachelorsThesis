import json
import pathlib

import invoke
import msgpack
import numpy as np
import requests

TARGET = pathlib.Path("data/mnist").resolve()


@invoke.task()
def fetch_training_data(ctx):
    """Download MNIST training data and normalize by 1 / 255."""
    (x_train, y_train), (x_test, y_test) = (data.astype("float32") / 255 for data in keras.datasets.mnist.load_data())
    ctx.run(f"mkdir -p {TARGET}")
    np.save(target / "x-train.npy", x_train)
    np.save(target / "y-train.npy", y_train)
    np.save(target / "x-test.npy", x_test)
    np.save(target / "y-test.npy", y_test)


@invoke.task()
def send_test_request(ctx, index=3):
    """Sends an HTTP test request to localhost:5555"""
    x_test = np.load(TARGET / "x-test.npy")
    y_test = np.load(TARGET / "y-test.npy")
    response = msgpack.unpackb(
        requests.post(
            "http://localhost:8000/api/classify/plain/",
            data=msgpack.packb({"image": x_test[index].reshape((784,)).tolist()}),
        ).content
    )
    print("Response:", json.dumps(response, indent=2))
    print(f"Prediction is {'correct' if response['prediction'] == y_test[index] else 'wrong'}")
