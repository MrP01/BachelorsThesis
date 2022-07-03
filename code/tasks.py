"""Tasks to run using `invoke [task-name]`"""
# pylint: disable=unused-argument
import json
import pathlib

import invoke
import msgpack
import numpy as np
import requests

TARGET = pathlib.Path(__file__).resolve().parent / "classifier" / "data" / "mnist"


@invoke.task()
def fetch_training_data(ctx):
    """Download MNIST training data and normalize by 1 / 255."""
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    ctx.run(f"mkdir -p {TARGET}")
    np.save(TARGET / "x-train.npy", x_train.astype("float32") / 255)
    np.save(TARGET / "y-train.npy", y_train)
    np.save(TARGET / "x-test.npy", x_test.astype("float32") / 255)
    np.save(TARGET / "y-test.npy", y_test)
    print(f"Saved MNIST data to {TARGET}")


@invoke.task()
def train(ctx, plot_plots=False):
    """Trains the Tensorflow Model and produces plots if asked to"""
    import network

    model, history = network.train()

    if plot_plots:
        from plots import plots

        x_test = np.load(TARGET / "x-test.npy")
        y_test = np.load(TARGET / "y-test.npy")

        plots.confusion_matrix(model, x_test, y_test)
        plots.plot_metric(history)


@invoke.task()
def send_test_request(ctx, index=3):
    """Sends an HTTP test request to localhost:5555"""
    x_test = np.load(TARGET / "x-test.npy")
    y_test = np.load(TARGET / "y-test.npy")
    vector = x_test[index].reshape((784,))
    response = msgpack.unpackb(
        requests.post(
            "http://localhost:8000/api/classify/plain/",
            data=msgpack.packb({"image": vector.tolist()}),
        ).content
    )
    print("Response:", json.dumps(response, indent=2))
    print(f"Prediction is {'correct' if response['prediction'] == y_test[index] else 'wrong'}")


@invoke.task()
def generate_secrets(ctx):
    """Creates a self-signed SSL certificate and key and puts them into secrets/"""
    SECRETS_DIR = pathlib.Path("secrets/").resolve()
    print(f"Putting secrets into: {SECRETS_DIR}")
    ctx.run(
        "openssl req -x509 -newkey rsa:4096 -nodes "
        f'-keyout {SECRETS_DIR / "fhe-classifier.key"} '
        f'-out {SECRETS_DIR / "fhe-classifier.cert"} '
        '-days 3650 -subj "/C=AT/ST=Styria/L=Springfield/O=IAIK/CN=www.example.com"'
    )


namespace = invoke.Collection()
namespace.add_task(fetch_training_data)
namespace.add_task(send_test_request)
namespace.add_task(generate_secrets)
namespace.add_task(train)

try:
    from plots import plots as theplots

    namespace.add_collection(invoke.Collection.from_module(theplots))
except ImportError:
    print("Not adding in plots collection")
