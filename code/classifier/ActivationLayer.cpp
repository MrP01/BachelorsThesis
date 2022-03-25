#include "Layer.h"
#include "Network.h"
#include <plog/Log.h>

Vector ActivationLayer::feedforward(Vector x) {
  return 0.54738 + 0.59579 * x + 0.090189 * xt::pow(x, 2) - 0.006137 * xt::pow(x, 3);
}

seal::Ciphertext multiplyPlain(seal::Ciphertext &x, double coeff, seal::RelinKeys relinKeys, seal::CKKSEncoder &encoder,
    seal::Evaluator &evaluator) {
  seal::Ciphertext result;
  seal::Plaintext plain_coeff;
  encoder.encode(coeff, x.parms_id(), x.scale(), plain_coeff); // use the same parameters as x!
  evaluator.multiply_plain(x, plain_coeff, result);            // the scale doubles
  evaluator.relinearize_inplace(result, relinKeys);            // relinearize after every multiplication?
  evaluator.rescale_to_next_inplace(result);                   // rescale down again (-40 bits?)
  PLOG(plog::debug) << "Scale after multiplication: " << log2(result.scale()) << " bits";
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
