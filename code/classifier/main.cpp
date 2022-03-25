#include <backward.hpp>
#include <csignal>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
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

auto neuralNet = new Network();
bool quit = false;

namespace backward {
backward::SignalHandling _signalHandler;
}

nlohmann::json handlePlainPredictionRequest(nlohmann::json request) {
  xt::xarray<double> input;
  xt::from_json(request["image"], input);
  PLOG(plog::debug) << input;
  assert(input.dimension() == 1);
  assert(input.shape()[0] == 784);
  // input.reshape({784});
  PLOG(plog::debug) << "Incoming data is valid, predicting ...";
  Vector result = neuralNet->predict(input);
  return nlohmann::json{
      {"prediction", neuralNet->interpretResult(result)},
      {"probabilities", neuralNet->interpretResultProbabilities(result)},
  };
}

nlohmann::json handleEncryptedPredictionRequest(nlohmann::json request) {
  PLOG(plog::debug) << "Incoming encrypted request";

  seal::RelinKeys relinKeys;
  seal::GaloisKeys galoisKeys;
  seal::Ciphertext ciphertext;

  nlohmann::json::binary_t binary = request["relinKeys"].get<nlohmann::json::binary_t>();
  PLOG(plog::debug) << "Decoded length: " << binary.size();
  PLOG(plog::debug) << "NeuralNet context poly mod degree: "
                    << neuralNet->context->key_context_data()->parms().poly_modulus_degree();
  std::stringstream dataStream = std::stringstream(std::string(binary.begin(), binary.end()));
  assert(seal::Serialization::compr_mode_default == seal::compr_mode_type::zstd);
  relinKeys.load(*neuralNet->context, dataStream);

  binary = request["galoisKeys"].get<nlohmann::json::binary_t>();
  dataStream = std::stringstream(std::string(binary.begin(), binary.end()));
  galoisKeys.load(*neuralNet->context, dataStream);

  binary = request["ciphertext"].get<nlohmann::json::binary_t>();
  dataStream = std::stringstream(std::string(binary.begin(), binary.end()));
  ciphertext.load(*neuralNet->context, dataStream);

  seal::Ciphertext result = neuralNet->predictEncrypted(ciphertext, relinKeys, galoisKeys);

  std::vector<uint8_t> byte_buffer(static_cast<size_t>(result.save_size()));
  result.save(reinterpret_cast<seal::seal_byte *>(byte_buffer.data()), byte_buffer.size());
  auto binaryResult = nlohmann::json::binary(byte_buffer);
  return nlohmann::json{
      {"result", binaryResult},
      // {"probabilities", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, // prediction and probabilities should be calculated
      // client-side
  };
}

auto msgpackRequestHandler(nlohmann::json (*handler)(nlohmann::json)) {
  return
      [=](const httplib::Request &request, httplib::Response &response, const httplib::ContentReader &contentReader) {
        std::string request_body;
        contentReader([&](const char *data, size_t data_length) {
          request_body.append(data, data_length);
          return true;
        });
        nlohmann::json request_json = nlohmann::json::from_msgpack(request_body);
        nlohmann::json response_json = handler(request_json);
        std::vector<uint8_t> serialized = nlohmann::json::to_msgpack(response_json);
        response.set_content(std::string(serialized.begin(), serialized.end()), "application/x-msgpack");
      };
}

void runServer() {
  httplib::Server server;
  server.Post("/api/classify/plain/", msgpackRequestHandler(handlePlainPredictionRequest));
  server.Post("/api/classify/encrypted/", msgpackRequestHandler(handleEncryptedPredictionRequest));

  server.set_exception_handler([](const httplib::Request &req, httplib::Response &res, std::exception &exception) {
    PLOG(plog::debug) << "Exception caught: " << exception.what();
    backward::StackTrace trace;
    trace.load_here();
    backward::Printer printer;
    printer.print(trace);
    res.status = 500;
    res.set_content(exception.what(), "text/plain");
  });
  server.set_logger([](const httplib::Request &req, const httplib::Response &res) {
    // prints log after the response was sent
    PLOG(plog::debug) << "[" << req.method << "] " << req.path << " " << res.status;
  });

  PLOG(plog::info) << "The server is running";
  server.listen("0.0.0.0", 8000);
}

double evaluateNetworkOnTestData() {
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  auto y_test = xt::load_npy<int>("data/mnist/y-test.npy");
  int N = x_test.shape()[0] * 0.3;
  x_test /= 255;

  int correct = 0, i = 0;
  for (auto iter = xt::axis_begin(x_test, 0); iter != xt::axis_end(x_test, 0); iter++) {
    Vector x = *iter;
    x.reshape({784});
    Vector result = neuralNet->predict(x);
    int prediction = neuralNet->interpretResult(result);
    // PLOG(plog::debug) << prediction << " | " << y_test[i];
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

double evaluateNetworkOnEncryptedTestData() {
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  auto y_test = xt::load_npy<int>("data/mnist/y-test.npy");
  int N = 20;
  x_test /= 255;
  x_test.reshape({x_test.shape(0), 784});
  seal::KeyGenerator keyGen(*neuralNet->context);
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
  seal::Encryptor encryptor(*neuralNet->context, publicKey);
  seal::CKKSEncoder encoder(*neuralNet->context);
  double scale = pow(2.0, 40);

  double mre_sum = 0;
  int correct_predictions = 0;
  for (size_t i = 0; i < N; i++) {
    auto some_x_test = xt::view(x_test, i, xt::all());
    auto some_x_test_vector = std::vector<double>(some_x_test.begin(), some_x_test.end());
    assert(some_x_test_vector.size() == 784);
    encoder.encode(some_x_test_vector, scale, plain);
    encryptor.encrypt(plain, encrypted);
    seal::Ciphertext result = neuralNet->predictEncrypted(encrypted, relinKeys, galoisKeys);
    seal::Decryptor decryptor(*neuralNet->context, keyGen.secret_key());
    decryptor.decrypt(result, plain_result);
    encoder.decode(plain_result, decoded_plain_result);
    Vector result_from_encrypted_method = xt::adapt(decoded_plain_result, {10});
    auto exact_result = neuralNet->predict(some_x_test);
    int prediction = neuralNet->interpretResult(result_from_encrypted_method);
    PLOG(plog::debug) << "The encrypted method result: " << result_from_encrypted_method;
    PLOG(plog::debug) << "For comparison, plain result: " << exact_result;
    PLOG(plog::debug) << "Relative errors: " << xt::abs((result_from_encrypted_method - exact_result) / exact_result);
    auto mre = xt::mean(xt::abs(result_from_encrypted_method - exact_result) / xt::amax(xt::abs(exact_result)));
    PLOG(plog::debug) << "Mean max-relative error: " << mre;
    if (prediction == y_test[i]) {
      PLOG(plog::info) << "--> Correctly predicted!!";
      correct_predictions++;
    } else {
      PLOG(plog::info) << "--> Incorrect prediction :(";
    }
    mre_sum += mre();
  }
  PLOG(plog::info) << "Average MRE: " << mre_sum / N;
  PLOG(plog::info) << "Accuracy: " << (double)correct_predictions / N;
  return (double)correct_predictions / N;
}

void shutdown(int signum) {
  PLOG(plog::info) << "Shutdown...";
  quit = true;
}

int main() {
  static plog::ColorConsoleAppender<plog::FuncMessageFormatter> consoleAppender;
  plog::init(plog::info, &consoleAppender);
  PLOG(plog::info) << "--- MNIST Neural Network Predictor ---";
  signal(SIGTERM, shutdown);

  //  neuralNet->addLayer(784, 128);
  //  neuralNet->addLayer(128, 10);
  auto w1 = xt::load_npy<float>("data/models/simple/w1.npy");
  auto b1 = xt::load_npy<float>("data/models/simple/b1.npy");
  auto w2 = xt::load_npy<float>("data/models/simple/w2.npy");
  auto b2 = xt::load_npy<float>("data/models/simple/b2.npy");

  neuralNet->init();
  neuralNet->addLayer(new DenseLayer(w1, b1));
  neuralNet->addLayer(new ActivationLayer());
  neuralNet->addLayer(new DenseLayer(w2, b2));

  runServer();
  // evaluateNetworkOnTestData();
  // evaluateNetworkOnEncryptedTestData();
  return 0;
}
