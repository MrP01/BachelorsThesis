#pragma once

#include <xtensor/xarray.hpp>
#include "seal/ciphertext.h"
#include "seal/evaluator.h"
#include "seal/ckks.h"

typedef xt::xarray<double> Matrix;
typedef xt::xarray<double> Vector;

class Network;

class Layer {
  friend Network;  // Network can access my properties

  protected:
    Network* parent = nullptr;

  public:
    Layer() = default;
    virtual Vector feedforward(Vector x) = 0;
    virtual void feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys, seal::CKKSEncoder &ckksEncoder, seal::Evaluator &evaluator) = 0;
};

class DenseLayer: public Layer {
  private:
    Matrix weights;
    Vector biases;

    // help properties for use by backpropagation algorithm
    // Matrix nablaW;
    // Vector nablaB;
    // out;
    // out_prime;

  public:
    DenseLayer(Matrix weights, Vector biases);
    void matmulDiagonal(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys, seal::CKKSEncoder &ckks_encoder, seal::Evaluator &evaluator);
    void multiplyCKKSBabystepGiantstep(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys, seal::CKKSEncoder &ckks_encoder, seal::Evaluator &evaluator);
    virtual Vector feedforward(Vector x);
    virtual void feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys, seal::CKKSEncoder &ckksEncoder, seal::Evaluator &evaluator);
};

class ActivationLayer: public Layer {
  public:
    ActivationLayer() = default;
    // static Vector activation(Vector x);
    // static Vector activationPrime(Vector x);

    virtual Vector feedforward(Vector x);
    virtual void feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys, seal::CKKSEncoder &ckksEncoder, seal::Evaluator &evaluator);
};
