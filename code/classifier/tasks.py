import pathlib

import invoke
import json
import mnist
import numpy as np
import requests
import msgpack


@invoke.task()
def fetch_training_data(ctx, target="data/mnist"):
    """Download MNIST training data"""
    x_train, y_train, x_test, y_test = (data.astype("float32") / 255 for data in mnist.mnist())
    target = pathlib.Path(target).resolve()
    ctx.run(f"mkdir -p {target}")
    np.save(target / "x-train.npy", x_train)
    np.save(target / "y-train.npy", y_train)
    np.save(target / "x-test.npy", x_test)
    np.save(target / "y-test.npy", y_test)


@invoke.task()
def send_test_request(ctx, index=3):
    """Sends an HTTP test request to localhost:5555"""
    x_train, y_train, x_test, y_test = (data.astype("float32") / 255 for data in mnist.mnist())
    response = msgpack.unpackb(
        requests.post(
            "http://localhost:8000/api/classify/plain/",
            data=msgpack.packb({"image": x_test[index].reshape((784,)).tolist()}),
        ).content
    )
    print("Response:", json.dumps(response, indent=2))
    print(f"Prediction is {'correct' if response['prediction'] == y_test[index] else 'wrong'}")
