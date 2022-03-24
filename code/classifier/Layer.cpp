#include "Layer.h"
#include "Network.h"
#include <algorithm>
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
  // to print it: std::copy(dot.begin(), dot.end(), std::ostream_iterator<float>(std::cout, ", "));
  return dot + biases;
}

void DenseLayer::feedforwardEncrypted(seal::Ciphertext &in_out, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys,
    seal::CKKSEncoder &ckksEncoder, seal::Evaluator &evaluator) {
  unsigned in_dimension = weights.shape(0);
  unsigned out_dimension = weights.shape(1);
  std::cout << "DenseLayer input scale: " << log2(in_out.scale()) << " bits" << std::endl;
  // Matrix zeroPaddedSquareWeights;
  // if (in_dimension > out_dimension)
  //   zeroPaddedSquareWeights = xt::pad(weights, {{0, in_dimension - out_dimension}, {0, 0}});
  matmulDiagonal(in_out, weights, galoisKeys, ckksEncoder, evaluator);
  std::cout << "DenseLayer output scale: " << log2(in_out.scale()) << " bits" << std::endl;
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

Vector ActivationLayer::feedforward(Vector x) {
  return 0.54738 + 0.59579 * x + 0.090189 * xt::pow(x, 2) - 0.006137 * xt::pow(x, 3);
}

void printCiphertextInternals(
    std::string name, seal::Ciphertext &x, seal::SEALContext *context, bool exact_scale = false) {
  std::cout << "> " << name << " scale: ";
  if (exact_scale)
    std::cout << std::fixed << x.scale();
  else
    std::cout << log2(x.scale()) << " bits";
  std::cout << "; parms chain index: " << context->get_context_data(x.parms_id())->chain_index()
            << "; size: " << x.size() << std::endl;
}

seal::Ciphertext multiplyPlain(seal::Ciphertext &x, double coeff, seal::RelinKeys relinKeys, seal::CKKSEncoder &encoder,
    seal::Evaluator &evaluator) {
  seal::Ciphertext result;
  seal::Plaintext plain_coeff;
  encoder.encode(coeff, x.parms_id(), x.scale(), plain_coeff); // use the same parameters as x!
  evaluator.multiply_plain(x, plain_coeff, result);            // the scale doubles
  evaluator.relinearize_inplace(result, relinKeys);            // relinearize after every multiplication?
  evaluator.rescale_to_next_inplace(result);                   // rescale down again (-40 bits?)
  std::cout << "Scale after multiplication: " << log2(result.scale()) << " bits" << std::endl;
  return result;
}

void addPlainInplace(seal::Ciphertext &x, double y, seal::CKKSEncoder &encoder, seal::Evaluator &evaluator) {
  seal::Plaintext plain_coeff;
  encoder.encode(y, x.parms_id(), x.scale(), plain_coeff);
  evaluator.add_plain_inplace(x, plain_coeff);
}

void addThreeInplace(seal::Ciphertext &in_out, seal::Ciphertext &a, seal::Ciphertext &b, seal::Evaluator &evaluator) {
  // the three ciphertexts have a different scale, so we need to lie to Microsoft SEAL
  // (according to the documentation)
  double new_scale = pow(2.0, round(log2(in_out.scale())));
  assert(abs(in_out.scale() - new_scale) < new_scale * 0.00001);
  assert(abs(a.scale() - new_scale) < new_scale * 0.00001);
  assert(abs(b.scale() - new_scale) < new_scale * 0.00001);
  in_out.scale() = new_scale;
  a.scale() = new_scale;
  b.scale() = new_scale;

  // now that they have the same scale, we can freely add them
  evaluator.add_inplace(in_out, a);
  evaluator.add_inplace(in_out, b);
}

void ActivationLayer::feedforwardEncrypted(seal::Ciphertext &x, seal::GaloisKeys &galoisKeys, seal::RelinKeys relinKeys,
    seal::CKKSEncoder &encoder, seal::Evaluator &evaluator) {
  printCiphertextInternals("ActivationLayer input", x, parent->context);

  seal::Ciphertext x2, x3;
  evaluator.multiply(x, x, x2);
  evaluator.relinearize_inplace(x2, relinKeys);
  evaluator.rescale_to_next_inplace(x2);
  evaluator.mod_switch_to_next_inplace(x); // parms of x² changed one level down -> adjust level of x as well
  assert(x.parms_id() == x2.parms_id());

  evaluator.multiply(x2, x, x3);
  evaluator.relinearize_inplace(x3, relinKeys);
  evaluator.rescale_to_next_inplace(x3);
  evaluator.mod_switch_to_next_inplace(x);
  evaluator.mod_switch_to_next_inplace(x2);
  assert(x.parms_id() == x2.parms_id() && x.parms_id() == x3.parms_id());

  printCiphertextInternals("x", x, parent->context);
  printCiphertextInternals("x²", x2, parent->context);
  printCiphertextInternals("x³", x3, parent->context);

  auto result = multiplyPlain(x, 0.59579, relinKeys, encoder, evaluator);
  auto result2 = multiplyPlain(x2, 0.090189, relinKeys, encoder, evaluator);
  auto result3 = multiplyPlain(x3, -0.006137, relinKeys, encoder, evaluator);
  printCiphertextInternals("c_1 * x", result, parent->context, true);
  printCiphertextInternals("c_2 * x²", result2, parent->context, true);
  printCiphertextInternals("c_3 * x³", result3, parent->context, true);

  addThreeInplace(result, result2, result3, evaluator);
  addPlainInplace(result, 0.54738, encoder, evaluator);

  printCiphertextInternals("ActivationLayer output", result, parent->context);
  x = result;
}
