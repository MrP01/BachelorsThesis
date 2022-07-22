#include "Layer.h"
#include "Network.h"
#include <algorithm>
#include <plog/Log.h>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

static std::map<size_t, std::pair<size_t, size_t>> preencoded_bsgs_parameters = {{784, {28, 28}}, {128, {16, 8}}};
enum MatMulImplementation DenseLayer::matmulMethod = MATMUL_HYBRID;

DenseLayer::DenseLayer(Matrix weights, Vector biases) : weights(weights), biases(biases) {
  in_dim = weights.shape(0);
  out_dim = weights.shape(1);
  assert(biases.size() == out_dim);
}

Vector DenseLayer::feedforward(Vector x) {
  Vector dot = xt::zeros<double>({weights.shape()[1]});
  for (size_t col = 0; col < weights.shape()[1]; col++) {
    double sum = 0;
    for (size_t row = 0; row < weights.shape()[0]; row++)
      sum += weights.at(row, col) * x[row];
    dot[col] = sum;
  }
  // to print it: std::copy(dot.begin(), dot.end(), std::ostream_iterator<float>(PLOG(plog::debug), ", "));
  return dot + biases;
}

void DenseLayer::feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys,
    seal::RelinKeys &relinKeys, seal::CKKSEncoder &encoder, seal::Evaluator &evaluator) {
  IF_PLOG(plog::debug) { printCiphertextInternals("DenseLayer input", in_out, parent->context); }
  if (matmulMethod == MATMUL_BSGS)
    matmulBabystepGiantstep(in_out, galoisKeys, encoder, evaluator);
  else if (matmulMethod == MATMUL_DIAGONAL_MOD)
    matmulDiagonalMod(in_out, galoisKeys, encoder, evaluator);
  else if (matmulMethod == MATMUL_HYBRID)
    matmulHybrid(in_out, galoisKeys, encoder, evaluator);

  preencodedBiases.scale() = in_out.scale(); // the true scales should be nearly identical, so we fake it
  evaluator.add_plain_inplace(in_out, preencodedBiases);
  IF_PLOG(plog::debug) { printCiphertextInternals("DenseLayer output", in_out, parent->context); }
}

void DenseLayer::prepare(
    seal::CKKSEncoder &encoder, seal::Evaluator &evaluator, seal::parms_id_type parms_id, double scale) {
  seal::parms_id_type next_parms_id = parent->context->get_context_data(parms_id)->next_context_data()->parms_id();
  PLOG(plog::warning) << "Preparing with " << parent->context->get_context_data(parms_id)->chain_index()
                      << ", and scale " << scale;

  prepareDiagonals(encoder, evaluator, parms_id, scale);
  prepareBabystepGiantstep(encoder, evaluator, parms_id, scale);
  encoder.encode(std::vector<double>(biases.begin(), biases.end()), next_parms_id, scale, preencodedBiases);
}

void DenseLayer::prepareDiagonals(
    seal::CKKSEncoder &encoder, seal::Evaluator &evaluator, seal::parms_id_type parms_id, double scale) {
  size_t n_diagonals = in_dim; // compute all there are
  preencodedDiagonals.reserve(n_diagonals);
  plainDiagonals.reserve(n_diagonals);
  for (auto offset = 0ULL; offset < n_diagonals; offset++) {
    std::vector<double> diag;
    diag.reserve(in_dim);
    for (auto j = 0ULL; j < in_dim; j++)
      diag.push_back(weights.at((j + offset) % in_dim, j % out_dim));
    plainDiagonals.push_back(xt::adapt(diag));

    seal::Plaintext encoded;
    encoder.encode(diag, scale, encoded);
    evaluator.mod_switch_to_inplace(encoded, parms_id);
    preencodedDiagonals.push_back(encoded);
  }
}

void DenseLayer::dotMultiplyDiagonals(seal::Ciphertext &in_out, seal::GaloisKeys &galois_keys,
    seal::CKKSEncoder &encoder, seal::Evaluator &evaluator, enum DiagonalCount count) {
  size_t slots = encoder.slot_count(); // = N/2 = 4096/2 = 2048
  size_t n_diagonals = (count == IN_DIM) ? in_dim : out_dim;
  assert(in_dim > out_dim);
  if (in_dim != slots && in_dim * 2 > slots)
    throw std::runtime_error("too little slots for matmul implementation!");

  Vector input;
  if (debuggingDecryptor != nullptr)
    input = getCiphertextValue(in_out, in_dim, debuggingDecryptor, encoder);

  // perform the actual multiplication
  seal::Ciphertext sum = in_out;
  Vector plain_sum = Vector(input);
  evaluator.multiply_plain_inplace(sum, preencodedDiagonals[0]);
  plain_sum *= plainDiagonals[0];
  if (debuggingDecryptor != nullptr) {
    PLOG(plog::debug) << plain_sum;
    getCiphertextValue(sum, in_dim, debuggingDecryptor, encoder);
  }
  PLOG(plog::debug) << "--- now the following offsets:";
  for (auto offset = 1ULL; offset < n_diagonals; offset++) {
    seal::Ciphertext tmp;
    evaluator.rotate_vector_inplace(in_out, 1, galois_keys);
    if (debuggingDecryptor != nullptr) {
      auto plainy = xt::roll(input, -offset);
      Vector ency = getCiphertextValue(in_out, in_dim, debuggingDecryptor, encoder);
      PLOG(plog::debug) << "------> rot-diff at offset " << offset << ": " << xt::sum(xt::square(plainy - ency));
    }
    evaluator.multiply_plain(in_out, preencodedDiagonals[offset], tmp);
    evaluator.add_inplace(sum, tmp);
    if (debuggingDecryptor != nullptr) {
      Vector temp = xt::roll(input, -offset) * plainDiagonals[offset];
      plain_sum += temp;
      Vector ency = getCiphertextValue(tmp, in_dim, debuggingDecryptor, encoder);
      PLOG(plog::debug) << "------> tmp-diff at offset " << offset << ": " << xt::sum(xt::square(temp - ency));
    }
  }
  in_out = sum;
  evaluator.rescale_to_next_inplace(in_out); // scale down once

  if (debuggingDecryptor != nullptr)
    PLOG(plog::debug) << plain_sum;
}

void DenseLayer::matmulDiagonalMod(
    seal::Ciphertext &in_out, seal::GaloisKeys &galois_keys, seal::CKKSEncoder &encoder, seal::Evaluator &evaluator) {
  // if (slots != in_dim) {
  //   PLOG(plog::debug) << "Adding the rotated input vector to itself...";
  //   seal::Ciphertext in_out_rot;
  //   evaluator.rotate_vector(in_out, -((size_t)in_dim), galois_keys, in_out_rot);
  //   evaluator.add_inplace(in_out, in_out_rot);
  // }
  dotMultiplyDiagonals(in_out, galois_keys, encoder, evaluator, IN_DIM);
}

void DenseLayer::matmulHybrid(
    seal::Ciphertext &in_out, seal::GaloisKeys &galois_keys, seal::CKKSEncoder &encoder, seal::Evaluator &evaluator) {
  std::clock_t start = clock();
  dotMultiplyDiagonals(in_out, galois_keys, encoder, evaluator, OUT_DIM);
  std::clock_t end = clock();
  PLOG(plog::info) << "dotMultiply took " << (double)(end - start) / CLOCKS_PER_SEC;

  // perform the rotate-and-sum algorithm
  seal::Ciphertext rotated = in_out; // makes a copy
  for (size_t chunk = 0; chunk < in_dim / out_dim; chunk++) {
    evaluator.rotate_vector_inplace(rotated, out_dim, galois_keys);
    evaluator.add_inplace(in_out, rotated);
  }
}

void DenseLayer::prepareBabystepGiantstep(
    seal::CKKSEncoder &encoder, seal::Evaluator &evaluator, seal::parms_id_type parms_id, double scale) {
  Matrix zeroPaddedWeights = xt::zeros<double>({in_dim, in_dim});
  xt::view(zeroPaddedWeights, xt::range(0, out_dim), xt::all()) = xt::transpose(weights);
  assert(zeroPaddedWeights.shape(0) == zeroPaddedWeights.shape(1));

  size_t slots = encoder.slot_count(); // = N/2 = 4096/2 = 2048
  if (in_dim != slots && in_dim * 2 > slots)
    throw std::runtime_error("too little slots for matmul implementation!");

  std::pair bsgs_parameters = preencoded_bsgs_parameters.at(in_dim);
  size_t bsgs_n1 = bsgs_parameters.first, bsgs_n2 = bsgs_parameters.second;
  if (bsgs_n1 * bsgs_n2 != in_dim)
    throw std::runtime_error("wrong bsgs parameters");
  PLOG(plog::debug) << "BSGS parameters: " << bsgs_n1 << ", " << bsgs_n2;

  // baby step giant step method preparation:
  preencodedBSGS.reserve(in_dim);
  for (auto i = 0ULL; i < in_dim; i++) {
    std::vector<double> diag;
    auto k = i / bsgs_n1;
    diag.reserve(in_dim + k * bsgs_n1);

    for (auto j = 0ULL; j < in_dim; j++)
      diag.push_back(zeroPaddedWeights.at(j, (i + j) % in_dim));
    if (k)
      std::rotate(diag.begin(), diag.begin() + diag.size() - k * bsgs_n1, diag.end());

    // prepare for non-full-packed rotations
    if (slots != in_dim) {
      for (size_t index = 0; index < k * bsgs_n1; index++) {
        diag.push_back(diag[index]);
        diag[index] = 0;
      }
    }

    seal::Plaintext row;
    encoder.encode(diag, scale, row);
    evaluator.mod_switch_to_inplace(row, parms_id);
    preencodedBSGS.push_back(row);
  }
}

void DenseLayer::matmulBabystepGiantstep(
    seal::Ciphertext &in_out, seal::GaloisKeys &galois_keys, seal::CKKSEncoder &encoder, seal::Evaluator &evaluator) {
  std::clock_t start = clock();
  // prepare for non-full-packed rotations
  if (encoder.slot_count() != in_dim) {
    seal::Ciphertext in_out_rot;
    evaluator.rotate_vector(in_out, -((size_t)in_dim), galois_keys, in_out_rot);
    evaluator.add_inplace(in_out, in_out_rot);
  }

  seal::Ciphertext temp;
  seal::Ciphertext outer_sum;
  seal::Ciphertext inner_sum;
  std::pair bsgs_parameters = preencoded_bsgs_parameters.at(in_dim);
  size_t bsgs_n1 = bsgs_parameters.first, bsgs_n2 = bsgs_parameters.second;

  // prepare rotations
  std::vector<seal::Ciphertext> rot;
  rot.resize(bsgs_n1);
  rot[0] = in_out;
  for (size_t j = 1; j < bsgs_n1; j++)
    evaluator.rotate_vector(rot[j - 1], 1, galois_keys, rot[j]);

  for (size_t k = 0; k < bsgs_n2; k++) {
    evaluator.multiply_plain(rot[0], preencodedBSGS[k * bsgs_n1], inner_sum);
    for (size_t j = 1; j < bsgs_n1; j++) {
      evaluator.multiply_plain(rot[j], preencodedBSGS[k * bsgs_n1 + j], temp);
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
  evaluator.rescale_to_next_inplace(in_out); // scale down once
  std::clock_t end = clock();
  PLOG(plog::info) << "BSGS mult took " << (double)(end - start) / CLOCKS_PER_SEC;
}
