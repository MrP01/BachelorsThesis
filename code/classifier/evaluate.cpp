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

class Evaluator {
 public:
  Network network;
  seal::GaloisKeys galoisKeys;
  seal::RelinKeys relinKeys;
  seal::Encryptor encryptor;
  seal::CKKSEncoder encoder;
  seal::Evaluator evaluator;
  seal::Decryptor decryptor;
  size_t keySize;
  xt::xarray<float> x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  xt::xarray<uint8_t> y_test = xt::load_npy<uint8_t>("data/mnist/y-test.npy");

  Evaluator(Network network_, seal::SecretKey secretKey, seal::PublicKey publicKey, seal::GaloisKeys galoisKeys,
      seal::RelinKeys &relinKeys)
      : network(network_), encryptor(*network_.context, publicKey, secretKey), encoder(*network_.context),
        decryptor(*network_.context, secretKey), evaluator(*network_.context), galoisKeys(galoisKeys),
        relinKeys(relinKeys) {
    x_test.reshape({x_test.shape(0), 784});
    PLOG(plog::info) << "Relin keys: " << relinKeys.save_size(seal::compr_mode_type::zstd);
    PLOG(plog::info) << "Galois keys: " << galoisKeys.save_size(seal::compr_mode_type::zstd);
    keySize = galoisKeys.save_size(seal::compr_mode_type::zstd) + relinKeys.save_size(seal::compr_mode_type::zstd);
  }

  void prepareLayers() {
    network.layers[0]->prepare(encoder, evaluator, network.context->first_parms_id(), SCALE); // chain index 5
    network.layers[2]->prepare(
        encoder, evaluator, network.context->last_context_data().get()->prev_context_data()->parms_id(), SCALE); // 1
  }

  double evaluateNetworkOnTestData(size_t N = 300) {
    int correct = 0, i = 0;
    for (auto iter = xt::axis_begin(x_test, 0); iter != xt::axis_end(x_test, 0); iter++) {
      Vector x = *iter, result = network.predict(x);
      int prediction = network.interpretResult(result);
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

  double evaluateNetworkOnEncryptedTestData(size_t N = 20) {
    seal::Plaintext plain;
    seal::Ciphertext encrypted;
    double mre_sum = 0;
    int correct_predictions = 0;
    std::string filename_base("data/ciphertext-visualisation/");
    for (size_t i = 0; i < N; i++) {
      auto some_x_test = xt::view(x_test, i, xt::all());
      auto some_x_test_vector = std::vector<double>(some_x_test.begin(), some_x_test.end());
      xt::dump_npy(filename_base + std::to_string(i) + "-plain.npy", some_x_test);
      assert(some_x_test_vector.size() == 784);
      encoder.encode(some_x_test_vector, SCALE, plain);
      encryptor.encrypt(plain, encrypted);
      auto visualisation = network.interpretCiphertextAsPixels(encrypted);
      xt::dump_npy(filename_base + std::to_string(i) + "-ciphertext.npy", visualisation);
      network.saveXArrayToPNG(filename_base + std::to_string(i) + "-ciphertext.png", visualisation);

      // seal::Ciphertext result = network.predictEncrypted(encrypted, relinKeys, galoisKeys);
      int index = 0;
      Vector plain = some_x_test;
      for (Layer *layer : network.layers) {
        PLOG(plog::debug) << "Feeding ciphertext through layer " << index++;

        plain = layer->feedforward(plain);
        PLOG(plog::debug) << "[Intermediate result]: exact: " << plain;

        IF_PLOG(plog::debug) { layer->debuggingDecryptor = &decryptor; }
        layer->feedforwardEncrypted(encrypted, galoisKeys, relinKeys, encoder, evaluator);
        printCiphertextInternals("Intermediate result", encrypted, network.context);
        Vector enc = getCiphertextValue(encrypted, plain.shape(0), &decryptor, encoder);
        PLOG(plog::debug) << "--> diff: " << xt::sum(xt::square(enc - plain));
        PLOG(plog::debug) << "----------------------------------------------------------------------------------------";
      }
      seal::Ciphertext result = encrypted;

      Vector result_from_encrypted_method = getCiphertextValue(result, 10, &decryptor, encoder);
      auto exact_result = network.predict(some_x_test);
      int prediction = network.interpretResult(result_from_encrypted_method);
      PLOG(plog::debug) << "Exact result:    " << exact_result;
      PLOG(plog::debug) << "Relative errors: " << xt::abs((result_from_encrypted_method - exact_result) / exact_result);
      double mre = meanMaxRelativeError(result_from_encrypted_method, exact_result);
      PLOG(plog::debug) << "Mean max-relative error: " << mre;
      if (prediction == y_test[i]) {
        PLOG(plog::info) << "--> Correctly predicted!!";
        correct_predictions++;
      } else {
        PLOG(plog::info) << "--> Incorrect prediction " << prediction << " (correct: " << y_test[i] << ")";
      }
      mre_sum += mre;
    }
    PLOG(plog::info) << "Average MRE: " << mre_sum / N;
    PLOG(plog::info) << "Accuracy: " << (double)correct_predictions / N;
    return (double)correct_predictions / N;
  }

  std::pair<double, double> runOnce(size_t i, bool symmetric = false) {
    seal::Plaintext plain;
    seal::Ciphertext encrypted;
    auto some_x_test = xt::view(x_test, i, xt::all());
    auto some_x_test_vector = std::vector<double>(some_x_test.begin(), some_x_test.end());

    std::clock_t start = clock();
    encoder.encode(some_x_test_vector, SCALE, plain);
    if (symmetric)
      encryptor.encrypt_symmetric(plain, encrypted);
    else
      encryptor.encrypt(plain, encrypted);
    size_t ciphertext_size = encrypted.save_size(seal::compr_mode_type::zstd);
    seal::Ciphertext result = network.predictEncrypted(encrypted, relinKeys, galoisKeys);
    ciphertext_size += result.save_size(seal::compr_mode_type::zstd);
    Vector output = getCiphertextValue(result, 10, &decryptor, encoder);
    std::clock_t end = clock();

    Vector exact_result = network.predict(some_x_test);
    double mre = meanMaxRelativeError(output, exact_result);
    int prediction = network.interpretResult(output);
    if (prediction == y_test[i])
      PLOG(plog::debug) << "correct";
    PLOG(plog::info) << "Runtime: " << (double)(end - start) / CLOCKS_PER_SEC
                     << "s \t ZStd ciphertext size: " << ciphertext_size
                     << ", total size with keys: " << (double)(ciphertext_size + keySize) / 1024 / 1024
                     << " MibiBytes \t MRE: " << mre;
    return std::pair<double, double>((double)(end - start) / CLOCKS_PER_SEC, mre);
  }

  void benchmark(bool symmetric = false) {
    size_t N = 3;
    PLOG(plog::info) << "Starting benchmark with " << COEFF_MODULUS_START_BITS << ", " << COEFF_MODULUS_MIDDLE_BITS
                     << ", poly mod degree " << POLY_MOD_DEGREE << " and matmul method " << DenseLayer::matmulMethod
                     << " and symmetric encryption: " << symmetric << " and seclevel " << (int)SECURITY_LEVEL;
    double totalRuntime = 0, totalMRE = 0;
    for (size_t i = 0; i < N; i++) {
      auto pair = runOnce(rand() % x_test.shape(0), symmetric);
      totalRuntime += pair.first;
      totalMRE += pair.second;
    }
    PLOG(plog::info) << "Average runtime over " << N << " runs: " << totalRuntime / N
                     << " s, Average MRE: " << totalMRE / N;
  }

  void fullBenchmark() {
    // DenseLayer::matmulMethod = MATMUL_DIAGONAL_MOD;
    // benchmark(true);
    DenseLayer::matmulMethod = MATMUL_HYBRID;
    benchmark(true);
    DenseLayer::matmulMethod = MATMUL_BSGS;
    benchmark(true);
  }
};

int main(int argc, char *argv[]) {
  static plog::ColorConsoleAppender<plog::FuncMessageFormatter> appender;
  plog::init(plog::info, &appender);
  xt::print_options::set_line_width(120);

  size_t evalPlain = 0, evalEnc = 0;
  if (argc >= 2)
    evalPlain = atol(argv[1]);
  if (argc >= 3)
    evalEnc = atol(argv[2]);

  Network network;
  network.init();
  network.loadDefaultModel();
  seal::KeyGenerator keyGen(*network.context);
  DenseLayer::matmulMethod = MATMUL_DIAGONAL_MOD;

  seal::PublicKey publicKey;
  seal::GaloisKeys galoisKeys;
  seal::RelinKeys relinKeys;
  std::clock_t keyGen_start = clock();
  keyGen.create_public_key(publicKey);
  keyGen.create_galois_keys(galoisKeys);
  keyGen.create_relin_keys(relinKeys);
  std::clock_t keyGen_end = clock();

  PLOG(plog::info) << "Key Generation time: " << (double)(keyGen_end - keyGen_start) / CLOCKS_PER_SEC;
  Evaluator evaluator(network, keyGen.secret_key(), publicKey, galoisKeys, relinKeys);
  evaluator.prepareLayers();
  PLOG(plog::info) << "keySize: " << evaluator.keySize;
  if (evalPlain)
    evaluator.evaluateNetworkOnTestData(evalPlain);
  if (evalEnc)
    evaluator.evaluateNetworkOnEncryptedTestData(evalEnc);
  if (!evalPlain && !evalEnc)
    evaluator.fullBenchmark();
  return 0;
}
