import invoke
import mnist


@invoke.task()
def fetch_training_data(ctx):
    """Download MNIST training data"""
    data = mnist.mnist()
    print(data)
