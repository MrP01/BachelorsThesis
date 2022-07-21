#pragma once
#include "utils.h"

#include <seal/ciphertext.h>
#include <seal/ckks.h>
#include <seal/decryptor.h>
#include <seal/evaluator.h>
#include <xtensor/xarray.hpp>

enum MatMulImplementation { MATMUL_DIAGONAL_MOD, MATMUL_HYBRID, MATMUL_BSGS };
enum DiagonalCount { IN_DIM, OUT_DIM };

class Network;

class Layer {
  friend Network; // Network can access my properties

 protected:
  Network *parent = nullptr;

 public:
  Layer() = default;
  virtual Vector feedforward(Vector x) = 0;
  virtual void feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys,
      seal::CKKSEncoder &ckksEncoder, seal::Evaluator &evaluator) = 0;
  seal::Decryptor *debuggingDecryptor = nullptr;
};

class DenseLayer : public Layer {
 private:
  Matrix weights;
  Vector biases;

  void dotMultiplyDiagonals(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys,
      seal::CKKSEncoder &encoder, seal::Evaluator &evaluator, enum DiagonalCount count);

 public:
  DenseLayer(Matrix weights, Vector biases);
  void matmulDiagonalMod(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys,
      seal::CKKSEncoder &ckks_encoder, seal::Evaluator &evaluator);
  void matmulHybrid(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys,
      seal::CKKSEncoder &ckks_encoder, seal::Evaluator &evaluator);
  void matmulBabystepGiantstep(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys,
      seal::CKKSEncoder &ckks_encoder, seal::Evaluator &evaluator);
  virtual Vector feedforward(Vector x);
  virtual void feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys,
      seal::CKKSEncoder &ckksEncoder, seal::Evaluator &evaluator);

  static enum MatMulImplementation matmulMethod;
};

class ActivationLayer : public Layer {
 public:
  ActivationLayer() = default;

  virtual Vector feedforward(Vector x);
  virtual void feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys,
      seal::CKKSEncoder &ckksEncoder, seal::Evaluator &evaluator);
};
