import tenseal

from .multiplications import *

ctx = tenseal.context(tenseal.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])


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
