#include "Layer.h"
#include "Network.h"
#include <algorithm>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>

int current_multiplication_level = 1;
int scale = 7;
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
  // Matrix zeroPaddedSquareWeights;
  // if (in_dimension > out_dimension)
  //   zeroPaddedSquareWeights = xt::pad(weights, {{0, in_dimension - out_dimension}, {0, 0}});
  matmulDiagonal(in_out, weights, galoisKeys, ckksEncoder, evaluator);
  // activationEncrypted(in_out, relinKeys, ckksEncoder, evaluator);
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
}

void DenseLayer::multiplyCKKSBabystepGiantstep(seal::Ciphertext &in_out, const Matrix &mat,
                                               seal::GaloisKeys &galois_keys, seal::CKKSEncoder &ckks_encoder,
                                               seal::Evaluator &evaluator) {
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

void ActivationLayer::feedforwardEncrypted(seal::Ciphertext &x1_encrypted, seal::GaloisKeys &galoisKeys,
                                           seal::RelinKeys relinKeys, seal::CKKSEncoder &encoder,
                                           seal::Evaluator &evaluator) {
  /*
  We create plaintexts for PI, 0.4, and 1 using an overload of CKKSEncoder::encode
  that encodes the given floating-point value to every slot in the vector.
  */
  seal::Plaintext plain_coeff3, plain_coeff2, plain_coeff1, plain_coeff0;
  encoder.encode(-0.006137, scale, plain_coeff3);
  encoder.encode(0.090189, scale, plain_coeff2);
  encoder.encode(0.59579, scale, plain_coeff1);
  encoder.encode(0.54738, scale, plain_coeff0);

  /*
  To compute x^3 we first compute x^2 and relinearize. However, the scale has
  now grown to 2^80.
  */
  seal::Ciphertext x2_encrypted;
  std::cout << "Compute x^2 and relinearize:" << std::endl;
  evaluator.square(x1_encrypted, x2_encrypted);
  evaluator.relinearize_inplace(x2_encrypted, relinKeys);
  std::cout << "    + Scale of x^2 before rescale: " << log2(x2_encrypted.scale()) << " bits" << std::endl;

  /*
  Now rescale; in addition to a modulus switch, the scale is reduced down by
  a factor equal to the prime that was switched away (40-bit prime). Hence, the
  new scale should be close to 2^40. Note, however, that the scale is not equal
  to 2^40: this is because the 40-bit prime is only close to 2^40.
  */
  std::cout << "Rescale x^2." << std::endl;
  evaluator.rescale_to_next_inplace(x2_encrypted);
  std::cout << "    + Scale of x^2 after rescale: " << log2(x2_encrypted.scale()) << " bits" << std::endl;

  /*
  Now x3_encrypted is at a different level than x1_encrypted, which prevents us
  from multiplying them to compute x^3. We could simply switch x1_encrypted to
  the next parameters in the modulus switching chain. However, since we still
  need to multiply the x^3 term with PI (plain_coeff3), we instead compute PI*x
  first and multiply that with x^2 to obtain PI*x^3. To this end, we compute
  PI*x and rescale it back from scale 2^80 to something close to 2^40.
  */
  seal::Ciphertext x3_encrypted(x2_encrypted); // TODO: is this a copy operation?
  std::cout << "Compute and rescale PI*x." << std::endl;
  seal::Ciphertext x1_encrypted_coeff3;
  evaluator.multiply_plain(x1_encrypted, plain_coeff3, x1_encrypted_coeff3);
  std::cout << "    + Scale of PI*x before rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << std::endl;
  evaluator.rescale_to_next_inplace(x1_encrypted_coeff3);
  std::cout << "    + Scale of PI*x after rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << std::endl;

  /*
  Multiply x2_encrypted by its corresponding coefficient plain_coeff2
  */
  evaluator.multiply_plain_inplace(x2_encrypted, plain_coeff2);
  // TODO: probably we need to rescale

  /*
  Since x3_encrypted and x1_encrypted_coeff3 have the same exact scale and use
  the same encryption parameters, we can multiply them together. We write the
  result to x3_encrypted, relinearize, and rescale. Note that again the scale
  is something close to 2^40, but not exactly 2^40 due to yet another scaling
  by a prime. We are down to the last level in the modulus switching chain.
  */
  std::cout << "Compute, relinearize, and rescale (PI*x)*x^2." << std::endl;
  evaluator.multiply_inplace(x3_encrypted, x1_encrypted_coeff3);
  evaluator.relinearize_inplace(x3_encrypted, relinKeys);
  std::cout << "    + Scale of PI*x^3 before rescale: " << log2(x3_encrypted.scale()) << " bits" << std::endl;
  evaluator.rescale_to_next_inplace(x3_encrypted);
  std::cout << "    + Scale of PI*x^3 after rescale: " << log2(x3_encrypted.scale()) << " bits" << std::endl;

  /*
  Next we compute the degree one term. All this requires is one multiply_plain
  with plain_coeff1. We overwrite x1_encrypted with the result.
  */
  std::cout << "Compute and rescale 0.4*x." << std::endl;
  evaluator.multiply_plain_inplace(x1_encrypted, plain_coeff1);
  std::cout << "    + Scale of 0.4*x before rescale: " << log2(x1_encrypted.scale()) << " bits" << std::endl;
  evaluator.rescale_to_next_inplace(x1_encrypted);
  std::cout << "    + Scale of 0.4*x after rescale: " << log2(x1_encrypted.scale()) << " bits" << std::endl;

  /*
  Now we would hope to compute the sum of all three terms. However, there is
  a serious problem: the encryption parameters used by all three terms are
  different due to modulus switching from rescaling.

  Encrypted addition and subtraction require that the scales of the inputs are
  the same, and also that the encryption parameters (parms_id) match. If there
  is a mismatch, Evaluator will throw an exception.
  */
  std::cout << std::endl;

  std::cout << "Parameters used by all three terms are different." << std::endl;
  std::cout << "  + Modulus chain index for x3_encrypted: "
            << parent->context->get_context_data(x3_encrypted.parms_id())->chain_index() << std::endl;
  std::cout << "  + Modulus chain index for x1_encrypted: "
            << parent->context->get_context_data(x1_encrypted.parms_id())->chain_index() << std::endl;
  std::cout << "  + Modulus chain index for plain_coeff0: "
            << parent->context->get_context_data(plain_coeff0.parms_id())->chain_index() << std::endl;
  std::cout << std::endl;

  /*
  Let us carefully consider what the scales are at this point. We denote the
  primes in coeff_modulus as P_0, P_1, P_2, P_3, in this order. P_3 is used as
  the special modulus and is not involved in rescalings. After the computations
  above the scales in ciphertexts are:

      - Product x^2 has scale 2^80 and is at level 2;
      - Product PI*x has scale 2^80 and is at level 2;
      - We rescaled both down to scale 2^80/P_2 and level 1;
      - Product PI*x^3 has scale (2^80/P_2)^2;
      - We rescaled it down to scale (2^80/P_2)^2/P_1 and level 0;
      - Product 0.4*x has scale 2^80;
      - We rescaled it down to scale 2^80/P_2 and level 1;
      - The contant term 1 has scale 2^40 and is at level 2.

  Although the scales of all three terms are approximately 2^40, their exact
  values are different, hence they cannot be added together.
  */
  std::cout << "The exact scales of all three terms are different:" << std::endl;
  // ios old_fmt(nullptr);
  // old_fmt.copyfmt(std::cout);
  // std::cout << fixed << setprecision(10);
  std::cout << "    + Exact scale in PI*x^3: " << x3_encrypted.scale() << std::endl;
  std::cout << "    + Exact scale in  0.4*x: " << x1_encrypted.scale() << std::endl;
  std::cout << "    + Exact scale in      1: " << plain_coeff0.scale() << std::endl;
  std::cout << std::endl;
  // std::cout.copyfmt(old_fmt);

  /*
  There are many ways to fix this problem. Since P_2 and P_1 are really close
  to 2^40, we can simply "lie" to Microsoft SEAL and set the scales to be the
  same. For example, changing the scale of PI*x^3 to 2^40 simply means that we
  scale the value of PI*x^3 by 2^120/(P_2^2*P_1), which is very close to 1.
  This should not result in any noticeable error.

  Another option would be to encode 1 with scale 2^80/P_2, do a multiply_plain
  with 0.4*x, and finally rescale. In this case we would need to additionally
  make sure to encode 1 with appropriate encryption parameters (parms_id).

  In this example we will use the first (simplest) approach and simply change
  the scale of PI*x^3 and 0.4*x to 2^40.
  */
  std::cout << "Normalize scales to 2^40." << std::endl;
  x3_encrypted.scale() = pow(2.0, 40);
  x1_encrypted.scale() = pow(2.0, 40);

  /*
  We still have a problem with mismatching encryption parameters. This is easy
  to fix by using traditional modulus switching (no rescaling). CKKS supports
  modulus switching just like the BFV scheme, allowing us to switch away parts
  of the coefficient modulus when it is simply not needed.
  */
  std::cout << "Normalize encryption parameters to the lowest level." << std::endl;
  seal::parms_id_type last_parms_id = x3_encrypted.parms_id();
  evaluator.mod_switch_to_inplace(x1_encrypted, last_parms_id);
  evaluator.mod_switch_to_inplace(plain_coeff0, last_parms_id);

  /*
  All three ciphertexts are now compatible and can be added.
  */
  std::cout << "Compute PI*x^3 + 0.4*x + 1." << std::endl;
  evaluator.add_inplace(x1_encrypted, x2_encrypted);
  evaluator.add_inplace(x1_encrypted, x3_encrypted);
  evaluator.add_plain_inplace(x1_encrypted, plain_coeff0);
  // return the result in x1_encrypted
}
