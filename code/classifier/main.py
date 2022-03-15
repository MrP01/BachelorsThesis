import numpy as np
import tenseal

ctx = tenseal.context(tenseal.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
w1 = np.load("data/models/simple/w1.npy")
b1 = np.load("data/models/simple/b1.npy")
w2 = np.load("data/models/simple/w2.npy")
b2 = np.load("data/models/simple/b2.npy")
x_test = np.load("data/mnist/x-test.npy") / 255
y_test = np.load("data/mnist/y-test.npy")

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
taylor_relu = lambda x: -0.006137 * x**3 + 0.090189 * x**2 + 0.59579 * x + 0.54738


def _matrix_diagonals(matrix: np.ndarray, offsets="out_dim"):
    out_dim, in_dim = matrix.shape
    offsets = out_dim if offsets == "out_dim" else in_dim
    return [[matrix[i % out_dim, (i + offset) % in_dim] for i in range(in_dim)] for offset in range(offsets)]


def matmul_diagonal(matrix: np.ndarray, vector):
    out_dim, in_dim = matrix.shape
    assert in_dim >= out_dim
    assert len(vector) == in_dim
    diagonals = _matrix_diagonals(matrix, "in_dim")
    sum_ = sum(np.roll(vector, -i) * diag for i, diag in enumerate(diagonals))
    return sum_[:out_dim]


def matmul_hybrid(matrix: np.ndarray, vector):
    out_dim, in_dim = matrix.shape
    assert in_dim >= out_dim
    assert in_dim % out_dim == 0
    assert len(vector) == in_dim
    diagonals = _matrix_diagonals(matrix, "out_dim")
    sum_ = sum(np.roll(vector, -i) * diag for i, diag in enumerate(diagonals))
    return sum(sum_[n * out_dim : (n + 1) * out_dim] for n in range(in_dim // out_dim))


def classify_plain(x):
    out_1 = taylor_relu(w1.T @ x + b1)
    out_2 = w2.T @ out_1 + b2
    return softmax(out_2)


def classify_encrypted(x):
    x_enc = tenseal.ckks_vector(ctx, x)
    out_1 = taylor_relu(x_enc.matmul(w1) + b1)
    out_2 = out_1.matmul(w2) + b2
    return softmax(out_2.decrypt())


def main():
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()


if __name__ == "__main__":
    main()
