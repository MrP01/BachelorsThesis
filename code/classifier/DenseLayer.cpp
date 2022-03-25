#include "Layer.h"
#include "Network.h"
#include <algorithm>
#include <plog/Log.h>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>

int current_multiplication_level = 1;
double scale = pow(2.0, 40);
int bsgs_n1 = 28, bsgs_n2 = 28; // product = matrix size, minimal sum if possible

DenseLayer::DenseLayer(Matrix weights, Vector biases) : weights(weights), biases(biases) {}

Vector DenseLayer::feedforward(Vector x) {
  Vector dot = xt::zeros<double>({weights.shape()[1]});
  for (uint64_t col = 0; col < weights.shape()[1]; col++) {
    double sum = 0;
    for (uint64_t row = 0; row < weights.shape()[0]; row++)
      sum += weights.at(row, col) * x[row];
    dot[col] = sum;
  }
  // to print it: std::copy(dot.begin(), dot.end(), std::ostream_iterator<float>(PLOG(plog::debug), ", "));
  return dot + biases;
}

void DenseLayer::feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys,
    seal::CKKSEncoder &encoder, seal::Evaluator &evaluator) {
  unsigned in_dimension = weights.shape(0);
  unsigned out_dimension = weights.shape(1);
  printCiphertextInternals("DenseLayer input", in_out, parent->context);

  // Matrix zeroPaddedSquareWeights;
  // if (in_dimension > out_dimension)
  //   zeroPaddedSquareWeights = xt::pad(weights, {{0, in_dimension - out_dimension}, {0, 0}});
  matmulDiagonal(in_out, weights, galoisKeys, encoder, evaluator);

  seal::Plaintext plain_biases;
  encoder.encode(std::vector<double>(biases.begin(), biases.end()), in_out.parms_id(), in_out.scale(), plain_biases);
  evaluator.add_plain_inplace(in_out, plain_biases);

  printCiphertextInternals("DenseLayer output", in_out, parent->context);
}

void DenseLayer::matmulDiagonal(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys,
    seal::CKKSEncoder &ckks_encoder, seal::Evaluator &evaluator) {
  int slots = ckks_encoder.slot_count(); // = N/2 = 4096/2 = 2048
  size_t in_dim = mat.shape(0);
  size_t out_dim = mat.shape(1);
  assert(in_dim > out_dim);
  if (in_dim != slots && in_dim * 2 > slots)
    throw std::runtime_error("too little slots for matmul implementation!");

  if (slots != in_dim) {
    seal::Ciphertext in_out_rot;
    evaluator.rotate_vector(in_out, -((int)in_dim), galois_keys, in_out_rot);
    evaluator.add_inplace(in_out, in_out_rot);
  }

  // diagonal method preparation:
  std::vector<seal::Plaintext> diagonals;
  diagonals.reserve(in_dim);
  for (auto offset = 0ULL; offset < in_dim; offset++) {
    std::vector<double> diag;
    diag.reserve(in_dim);
    for (auto j = 0ULL; j < in_dim; j++)
      diag.push_back(mat.at((j + offset) % in_dim, j % out_dim));
    seal::Plaintext encoded;
    ckks_encoder.encode(diag, scale, encoded);
    if (current_multiplication_level != 0)
      evaluator.mod_switch_to_inplace(encoded, in_out.parms_id());
    diagonals.push_back(encoded);
  }

  seal::Ciphertext sum = in_out;
  evaluator.multiply_plain_inplace(sum, diagonals[0]);
  for (auto i = 1ULL; i < in_dim; i++) {
    seal::Ciphertext tmp;
    evaluator.rotate_vector_inplace(in_out, 1, galois_keys);
    evaluator.multiply_plain(in_out, diagonals[i], tmp);
    evaluator.add_inplace(sum, tmp);
  }
  in_out = sum;
  evaluator.rescale_to_next_inplace(in_out); // scale down once
}

void DenseLayer::multiplyCKKSBabystepGiantstep(seal::Ciphertext &in_out, const Matrix &mat,
    seal::GaloisKeys &galois_keys, seal::CKKSEncoder &ckks_encoder, seal::Evaluator &evaluator) {
  int slots = ckks_encoder.slot_count(); // = N/2 = 4096/2 = 2048
  size_t matrix_dim = mat.shape()[0];
  if (matrix_dim != slots && matrix_dim * 2 > slots)
    throw std::runtime_error("too little slots for matmul implementation!");

  if (bsgs_n1 * bsgs_n2 != matrix_dim)
    throw std::runtime_error("wrong bsgs parameters");

  // baby step giant step method preparation:
  std::vector<seal::Plaintext> matrix;
  matrix.reserve(matrix_dim);
  for (auto i = 0ULL; i < matrix_dim; i++) {
    std::vector<double> diag;
    auto k = i / bsgs_n1;
    diag.reserve(matrix_dim + k * bsgs_n1);

    for (auto j = 0ULL; j < matrix_dim; j++)
      diag.push_back(mat.at(j, (i + j) % matrix_dim));
    // rotate:
    if (k)
      std::rotate(diag.begin(), diag.begin() + diag.size() - k * bsgs_n1, diag.end());

    // prepare for non-full-packed rotations
    if (slots != matrix_dim) {
      for (uint64_t index = 0; index < k * bsgs_n1; index++) {
        diag.push_back(diag[index]);
        diag[index] = 0;
      }
    }

    seal::Plaintext row;
    ckks_encoder.encode(diag, scale, row);
    if (current_multiplication_level != 0)
      evaluator.mod_switch_to_inplace(row, in_out.parms_id());
    matrix.push_back(row);
  }

  // prepare for non-full-packed rotations
  if (slots != matrix_dim) {
    seal::Ciphertext in_out_rot;
    evaluator.rotate_vector(in_out, -((int)matrix_dim), galois_keys, in_out_rot);
    evaluator.add_inplace(in_out, in_out_rot);
  }

  seal::Ciphertext temp;
  seal::Ciphertext outer_sum;
  seal::Ciphertext inner_sum;

  // prepare rotations
  std::vector<seal::Ciphertext> rot;
  rot.resize(bsgs_n1);
  rot[0] = in_out;
  for (uint64_t j = 1; j < bsgs_n1; j++)
    evaluator.rotate_vector(rot[j - 1], 1, galois_keys, rot[j]);

  for (uint64_t k = 0; k < bsgs_n2; k++) {
    evaluator.multiply_plain(rot[0], matrix[k * bsgs_n1], inner_sum);
    for (uint64_t j = 1; j < bsgs_n1; j++) {
      evaluator.multiply_plain(rot[j], matrix[k * bsgs_n1 + j], temp);
      evaluator.add_inplace(inner_sum, temp);
    }
    if (!k)
      outer_sum = inner_sum;
    else {
      evaluator.rotate_vector_inplace(inner_sum, k * bsgs_n1, galois_keys);
      evaluator.add_inplace(outer_sum, inner_sum);
    }
  }
  in_out = outer_sum;
}