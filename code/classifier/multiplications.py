"""Pure Python implementation of some of the matrix multiplication methods"""
import numpy as np

w1 = np.load("data/models/simple/w1.npy")
b1 = np.load("data/models/simple/b1.npy")
w2 = np.load("data/models/simple/w2.npy")
b2 = np.load("data/models/simple/b2.npy")
x_test = np.load("data/mnist/x-test.npy")
y_test = np.load("data/mnist/y-test.npy")
W1 = np.pad(w1, np.array(((0, 0), (0, 784 - 128))))

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
taylor_relu = lambda x: -0.006137 * x**3 + 0.090189 * x**2 + 0.59579 * x + 0.54738


def _matrix_diagonals(matrix: np.ndarray, offsets="out_dim"):
    out_dim, in_dim = matrix.shape
    offsets = out_dim if offsets == "out_dim" else in_dim
    return [[matrix[i % out_dim, (i + offset) % in_dim] for i in range(in_dim)] for offset in range(offsets)]


def matmul_diagonal(matrix: np.ndarray, vector):
    """Diagonal matmul method"""
    out_dim, in_dim = matrix.shape
    assert in_dim >= out_dim
    assert len(vector) == in_dim
    diagonals = _matrix_diagonals(matrix, "in_dim")
    sum_ = np.sum([np.roll(vector, -i) * diag for i, diag in enumerate(diagonals)], axis=0)
    return sum_[:out_dim]


def matmul_hybrid(matrix: np.ndarray, vector):
    """Hybrid matmul method (non-square matrices)"""
    out_dim, in_dim = matrix.shape
    assert in_dim >= out_dim
    assert in_dim % out_dim == 0
    assert len(vector) == in_dim
    diagonals = _matrix_diagonals(matrix, "out_dim")
    sum_ = np.sum([np.roll(vector, -i) * diag for i, diag in enumerate(diagonals)], axis=0)
    return sum(sum_[n * out_dim : (n + 1) * out_dim] for n in range(in_dim // out_dim))


def matmul_bsgs(matrix: np.ndarray, vector, t1, t2):
    """Babystep-Giantstep matmul method"""
    assert matrix.shape[0] == matrix.shape[1] == t1 * t2
    diagonals = _matrix_diagonals(matrix)
    diagp = lambda offset: np.roll(diagonals[offset], (offset // t1) * t1)
    return np.sum(
        [np.roll(sum(diagp(k * t1 + j) * np.roll(vector, -j) for j in range(t1)), -k * t1) for k in range(t2)], axis=0
    )


def classify_plain(x):
    """Classify a plain input vector x using the trained neural network"""
    out_1 = taylor_relu(w1.T @ x + b1)
    out_2 = w2.T @ out_1 + b2
    return softmax(out_2)


def main():
    """Run all"""
    v = np.random.randint(10, size=(784,))
    assert matmul_bsgs(W1.T, v, 28, 28)[:128] == w1.T @ v


if __name__ == "__main__":
    main()
