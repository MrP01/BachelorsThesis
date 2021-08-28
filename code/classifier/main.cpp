#include <backward.hpp>
#include <cppcodec/base64_rfc4648.hpp>
#include <csignal>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xjson.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>

#include "Network.h"

auto neuralNet = new Network();
bool quit = false;

namespace backward {
backward::SignalHandling sh;
}

nlohmann::json handlePlainPredictionRequest(nlohmann::json request) {
  xt::xarray<double> input;
  xt::from_json(request["image"], input);
  assert(input.dimension() == 1);
  assert(input.shape()[0] == 784);
  // input.reshape({784});
  std::cout << "Incoming data is valid, predicting ..." << std::endl;
  Vector result = neuralNet->predict(input);
  return nlohmann::json{
      {"prediction", neuralNet->interpretResult(result)},
      {"probabilities", neuralNet->interpretResultProbabilities(result)},
  };
}

nlohmann::json handleEncryptedPredictionRequest(nlohmann::json request) {
  std::cout << "Incoming encrypted request" << std::endl;

  seal::RelinKeys relinKeys;
  seal::GaloisKeys galoisKeys;
  seal::Evaluator evaluator();

  nlohmann::json::binary_t decoded = request["relinKeys"].get<nlohmann::json::binary_t>();
  std::cout << "Decoded length: " << decoded.size() << std::endl;
  std::cout << "NeuralNet context poly mod degree: " << neuralNet->context->key_context_data()->parms().poly_modulus_degree() << std::endl;
  std::stringstream dataStream = std::stringstream(std::string(decoded.begin(), decoded.end()));
  assert(seal::Serialization::compr_mode_default == seal::compr_mode_type::zstd);
  relinKeys.load(*neuralNet->context, dataStream);

  return nlohmann::json{
      {"prediction", 33},
      {"probabilities", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
  };
}

auto msgpackRequestHandler(nlohmann::json (*handler)(nlohmann::json)) {
  return [=](const httplib::Request &request, httplib::Response &response, const httplib::ContentReader &contentReader) {
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
    std::cout << "Exception caught: " << exception.what() << std::endl;
    res.status = 500;
    res.set_content(exception.what(), "text/plain");
  });

  server.set_logger([](const httplib::Request &req, const httplib::Response &res) { std::cout << "[" << req.method << "] " << req.path << " " << res.status << std::endl; });

  std::cout << "The server is running" << std::endl;
  server.listen("0.0.0.0", 8000);
}

double evaluateNetworkOnTestData() {
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  auto y_test = xt::load_npy<int>("data/mnist/y-test.npy");
  int N = x_test.shape()[0];
  // x_test.reshape({N, 784});

  int correct = 0, i = 0;
  for (auto iter = xt::axis_begin(x_test, 0); iter != xt::axis_end(x_test, 0); iter++) {
    Vector x = *iter;
    x.reshape({784});
    Vector result = neuralNet->predict(x);
    int prediction = neuralNet->interpretResult(result);
    // std::cout << prediction << " | " << y_test[i] << std::endl;
    if (i % 12 == 0)
      std::cout << i << " / " << N << "\r" << std::flush;
    if (prediction == y_test[i])
      correct++;
    i++;
  }
  std::cout << std::endl;
  double accuracy = (double)correct / (double)N;
  std::cout << "Correct: " << correct << " out of " << N << std::endl;
  std::cout << "Model accuracy: " << accuracy << std::endl;
  return accuracy;
}

void shutdown(int signum) {
  std::cout << "Shutdown..." << std::endl;
  quit = true;
}

int main() {
  std::cout << "--- MNIST Neural Network Predictor ---" << std::endl;
  signal(SIGTERM, shutdown);

  //  neuralNet->addLayer(784, 128);
  //  neuralNet->addLayer(128, 10);
  auto w1 = xt::load_npy<float>("data/models/simple/w1.npy");
  auto b1 = xt::load_npy<float>("data/models/simple/b1.npy");
  auto w2 = xt::load_npy<float>("data/models/simple/w2.npy");
  auto b2 = xt::load_npy<float>("data/models/simple/b2.npy");

  neuralNet->init();
  neuralNet->addLayer(new Layer(w1, b1));
  neuralNet->addLayer(new Layer(w2, b2));

  runServer();
  return 0;
}
