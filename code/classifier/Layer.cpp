#include "Layer.h"
#include "Network.h"
#include <algorithm>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xmath.hpp>

int current_multiplication_level = 1;
int scale = 7;
int bsgs_n1 = 28, bsgs_n2 = 28; // product = matrix size, minimal sum if possible

Layer::Layer(Matrix weights, Vector biases) : weights(weights), biases(biases) {}

Vector Layer::feedforward(Vector x) {
  // std::cout << x.shape()[0] << " - " << weights.shape()[0] << weights.shape()[1] << std::endl;
  // std::cout << xt::view(x) << std::endl;
  Vector dot = xt::zeros<double>({weights.shape()[1]});
  int i = 0;
  for (auto iter = xt::axis_begin(weights, 1); iter != xt::axis_end(weights, 1); iter++) {
    dot[i] = xt::sum((*iter) * x)();
    i++;
  }
  // std::copy(dot.begin(), dot.end(), std::ostream_iterator<float>(std::cout, ", "));
  return activation(dot + biases);
}

void Layer::multiplyCKKS(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys, seal::CKKSEncoder *ckks_encoder, seal::Evaluator &evaluator) {
  // int slots = parent->context->getSlots();  // TODO: = N/2 = 4096/2 = 2048
  int slots = 2048;
  size_t matrix_dim = mat.size();
  if (matrix_dim != slots && matrix_dim * 2 > slots)
    throw std::runtime_error("too little slots for matmul implementation!");

  if (slots != matrix_dim) {
    seal::Ciphertext in_out_rot;
    evaluator.rotate_vector(in_out, -((int)matrix_dim), galois_keys, in_out_rot);
    evaluator.add_inplace(in_out, in_out_rot);
  }

  // diagonal method preperation:
  std::vector<seal::Plaintext> matrix;
  matrix.reserve(matrix_dim);
  for (auto i = 0ULL; i < matrix_dim; i++) {
    std::vector<double> diag;
    diag.reserve(matrix_dim);
    for (auto j = 0ULL; j < matrix_dim; j++) {
      diag.push_back(mat.at((i + j) % matrix_dim, j));
    }
    seal::Plaintext row;
    ckks_encoder->encode(diag, scale, row);
    if (current_multiplication_level != 0)
      evaluator.mod_switch_to_inplace(row, in_out.parms_id());
    matrix.push_back(row);
  }

  seal::Ciphertext sum = in_out;
  evaluator.multiply_plain_inplace(sum, matrix[0]);
  for (auto i = 1ULL; i < matrix_dim; i++) {
    seal::Ciphertext tmp;
    evaluator.rotate_vector_inplace(in_out, 1, galois_keys);
    try {
      evaluator.multiply_plain(in_out, matrix[i], tmp);
      evaluator.add_inplace(sum, tmp);
    } catch (std::logic_error &ex) {
      // ignore transparent ciphertext
    }
  }
  in_out = sum;
}

void Layer::multiplyCKKSBabystepGiantstep(seal::Ciphertext &in_out, const Matrix &mat, seal::GaloisKeys &galois_keys, seal::CKKSEncoder *ckks_encoder, seal::Evaluator &evaluator) {
  int slots = 2048; // TODO: getSlots()
  size_t matrix_dim = mat.size();

  if (matrix_dim != slots && matrix_dim * 2 > slots)
    throw std::runtime_error("too little slots for matmul implementation!");

  if (bsgs_n1 * bsgs_n2 != matrix_dim)
    throw std::runtime_error("wrong bsgs parameters");

  // baby step giant step method preperation:
  std::vector<seal::Plaintext> matrix;
  matrix.reserve(matrix_dim);
  for (auto i = 0ULL; i < matrix_dim; i++) {
    std::vector<double> diag;
    auto k = i / bsgs_n1;
    diag.reserve(matrix_dim + k * bsgs_n1);

    for (auto j = 0ULL; j < matrix_dim; j++) {
      diag.push_back(mat.at((i + j) % matrix_dim, j));
    }
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
    ckks_encoder->encode(diag, scale, row);
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
      try {
        evaluator.multiply_plain(rot[j], matrix[k * bsgs_n1 + j], temp);
        evaluator.add_inplace(inner_sum, temp);
      } catch (std::logic_error &ex) {
        // ignore transparent ciphertext
      }
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

Vector Layer::activation(Vector x) { return 0.54738 + 0.59579 * x + 0.090189 * xt::pow(x, 2) - 0.006137 * xt::pow(x, 3); }
