#include <backward.hpp>
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
auto y_test = xt::load_npy<int>("data/mnist/y-test.npy");

namespace backward {
backward::SignalHandling _signalHandler;
}

double evaluateNetworkOnTestData(int N = 300) {
  int correct = 0, i = 0;
  for (auto iter = xt::axis_begin(x_test, 0); iter != xt::axis_end(x_test, 0); iter++) {
    Vector x = *iter, result = neuralNet.predict(x);
    int prediction = neuralNet.interpretResult(result);
    if (i % 12 == 0)
      PLOG(plog::debug) << i << " / " << N << "\r" << std::flush;
    if (i > N)
      break;
    if (prediction == y_test[i])
      correct++;
    i++;
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
  seal::Plaintext plain_result;
  std::vector<double> decoded_plain_result;
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
    seal::Ciphertext result = neuralNet.predictEncrypted(encrypted, relinKeys, galoisKeys);
    seal::Decryptor decryptor(*neuralNet.context, keyGen.secret_key());
    decryptor.decrypt(result, plain_result);
    encoder.decode(plain_result, decoded_plain_result);
    Vector result_from_encrypted_method = xt::adapt(decoded_plain_result, {10});
    auto exact_result = neuralNet.predict(some_x_test);
    int prediction = neuralNet.interpretResult(result_from_encrypted_method);
    PLOG(plog::debug) << "The encrypted method result: " << result_from_encrypted_method;
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
  x_test /= 255;
  x_test.reshape({x_test.shape(0), 784});

  neuralNet.init();
  neuralNet.loadDefaultModel();
  evaluateNetworkOnTestData();
  evaluateNetworkOnEncryptedTestData();
  return 0;
}
