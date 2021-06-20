#pragma once

#include <xtensor/xarray.hpp>
#include "seal/ciphertext.h"
#include "seal/evaluator.h"
#include "seal/ckks.h"

typedef xt::xarray<double> Matrix;
typedef xt::xarray<double> Vector;

class Layer {
  private:
    Matrix weights;
    Vector biases;

    // help properties for use by backpropagation algorithm
    // Matrix nablaW;
    // Vector nablaB;
    // out;
    // out_prime;

    static Vector activation(Vector x);
    // static Vector activationPrime(Vector x);

  public:
    Layer(Matrix weights, Vector biases);
    Vector feedforward(Vector x);
    void multiplyCKKS(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys, seal::CKKSEncoder* ckks_encoder, seal::Evaluator &evaluator);
    void multiplyCKKSBabystepGiantstep(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys, seal::CKKSEncoder* ckks_encoder, seal::Evaluator &evaluator);
};
