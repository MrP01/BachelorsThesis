#include <iostream>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/FuncMessageFormatter.h>
#include <plog/Log.h>
#include <seal/decryptor.h>
#include <seal/encryptor.h>
#include <seal/keygenerator.h>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xjson.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>

#include "Network.h"

Network neuralNet;
auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
auto y_test = xt::load_npy<uint8_t>("data/mnist/y-test.npy");

double evaluateNetworkOnTestData(int N = 300) {
  int correct = 0, i = 0;
  for (auto iter = xt::axis_begin(x_test, 0); iter != xt::axis_end(x_test, 0); iter++) {
    Vector x = *iter, result = neuralNet.predict(x);
    int prediction = neuralNet.interpretResult(result);
    if (i % 12 == 0)
      PLOG(plog::debug) << i << " / " << N << "\r" << std::flush;
    if (prediction == y_test[i])
      correct++;
    if (++i >= N)
      break;
  }
  PLOG(plog::debug);
  double accuracy = (double)correct / (double)N;
  PLOG(plog::debug) << "Correct: " << correct << " out of " << N;
  PLOG(plog::debug) << "Model accuracy: " << accuracy;
  return accuracy;
}

double evaluateNetworkOnEncryptedTestData(int N = 20) {
  seal::KeyGenerator keyGen(*neuralNet.context);
  seal::PublicKey publicKey;
  seal::GaloisKeys galoisKeys;
  seal::RelinKeys relinKeys;
  seal::Plaintext plain;
  seal::Ciphertext encrypted;
  keyGen.create_public_key(publicKey);
  keyGen.create_galois_keys(galoisKeys);
  keyGen.create_relin_keys(relinKeys);
  seal::Encryptor encryptor(*neuralNet.context, publicKey);
  seal::CKKSEncoder encoder(*neuralNet.context);
  double scale = pow(2.0, 40);

  double mre_sum = 0;
  int correct_predictions = 0;
  for (size_t i = 0; i < N; i++) {
    auto some_x_test = xt::view(x_test, i, xt::all());
    auto some_x_test_vector = std::vector<double>(some_x_test.begin(), some_x_test.end());
    assert(some_x_test_vector.size() == 784);
    encoder.encode(some_x_test_vector, scale, plain);
    encryptor.encrypt(plain, encrypted);
    seal::Decryptor decryptor(*neuralNet.context, keyGen.secret_key());

    // seal::Ciphertext result = neuralNet.predictEncrypted(encrypted, relinKeys, galoisKeys);
    seal::Evaluator evaluator(*neuralNet.context);
    int index = 0;
    Vector plain = some_x_test;
    for (Layer *layer : neuralNet.layers) {
      PLOG(plog::debug) << "Feeding ciphertext through layer " << index++;

      plain = layer->feedforward(plain);
      PLOG(plog::debug) << "[Intermediate result]: exact: " << plain;

      if (layer == neuralNet.layers[2])
        layer->debuggingDecryptor = &decryptor;
      layer->feedforwardEncrypted(encrypted, galoisKeys, relinKeys, encoder, evaluator);
      printCiphertextInternals("Intermediate result", encrypted, neuralNet.context);
      Vector enc = printCiphertextValue(encrypted, plain.shape(0), &decryptor, encoder);
      PLOG(plog::debug) << "--> diff: " << xt::sum(xt::square(enc - plain));
      PLOG(plog::info) << "-------------------------------------------------------------------------------------------";
    }
    seal::Ciphertext result = encrypted;

    Vector result_from_encrypted_method = printCiphertextValue(result, 10, &decryptor, encoder);
    auto exact_result = neuralNet.predict(some_x_test);
    int prediction = neuralNet.interpretResult(result_from_encrypted_method);
    PLOG(plog::debug) << "For comparison, plain result: " << exact_result;
    PLOG(plog::debug) << "Relative errors: " << xt::abs((result_from_encrypted_method - exact_result) / exact_result);
    auto mre = xt::mean(xt::abs(result_from_encrypted_method - exact_result) / xt::amax(xt::abs(exact_result)));
    PLOG(plog::debug) << "Mean max-relative error: " << mre;
    if (prediction == y_test[i]) {
      PLOG(plog::info) << "--> Correctly predicted!!";
      correct_predictions++;
    } else {
      PLOG(plog::info) << "--> Incorrect prediction " << prediction << " (correct: " << y_test[i] << ")";
    }
    mre_sum += mre();
  }
  PLOG(plog::info) << "Average MRE: " << mre_sum / N;
  PLOG(plog::info) << "Accuracy: " << (double)correct_predictions / N;
  return (double)correct_predictions / N;
}

int main() {
  static plog::ColorConsoleAppender<plog::FuncMessageFormatter> appender;
  plog::init(plog::debug, &appender);
  xt::print_options::set_line_width(120);
  x_test.reshape({x_test.shape(0), 784});

  neuralNet.init();
  neuralNet.loadDefaultModel();
  // evaluateNetworkOnTestData(10);
  evaluateNetworkOnEncryptedTestData(1);
  return 0;
}
