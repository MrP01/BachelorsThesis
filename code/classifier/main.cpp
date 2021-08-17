#include <backward.hpp>
#include <cppcodec/base64_rfc4648.hpp>
#include <csignal>
#include <iostream>
#include <nlohmann/json.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xjson.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>
#include <zmq.hpp>

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
      {"probabilites", neuralNet->interpretResultProbabilities(result)},
  };
}

nlohmann::json handleEncryptedPredictionRequest(nlohmann::json request) {
  using base64 = cppcodec::base64_rfc4648;

  seal::RelinKeys relinKeys;
  seal::GaloisKeys galoisKeys;
  seal::Evaluator evaluator();

  std::vector<uint8_t> decoded = base64::decode(request["relinKeys"].get<std::string>());
  std::stringstream dataStream(std::string(decoded.begin(), decoded.end()));
  relinKeys.load(*neuralNet->context, dataStream);

  decoded = base64::decode(request["galoisKeys"].get<std::string>());
  dataStream = std::stringstream(std::string(decoded.begin(), decoded.end()));
  galoisKeys.load(*neuralNet->context, dataStream);

  return nlohmann::json{};
}

nlohmann::json handleRequest(nlohmann::json request) {
  nlohmann::json response;
  if (request["action"].get<std::string>() == "predict_plain")
    response = handlePlainPredictionRequest(request);
  else if (request["action"].get<std::string>() == "predict_encrypted")
    response = handleEncryptedPredictionRequest(request);
  std::cout << "reply: " << response << std::endl;
  return response;
}

zmq::message_t handleMessage(zmq::message_t &request) {
  nlohmann::json request_json;
  try {
    request_json = nlohmann::json::parse(request.to_string());
  } catch (nlohmann::json::parse_error &ex) {
    std::cerr << "parse error at byte " << ex.byte << std::endl;
    return zmq::message_t("parse_error", 12);
  }
  nlohmann::json response_json = handleRequest(request_json);
  std::string serialized = response_json.dump();
  return zmq::message_t(serialized.c_str(), serialized.length());
}

void runServer() {
  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_REP);
  socket.bind("tcp://*:5555");
  std::cout << "The server is running" << std::endl;

  while (!quit) {
    try {
      zmq::message_t request;
      auto sent = socket.recv(request);
      std::cout << "Handling request ..." << std::endl;
      zmq::message_t reply = handleMessage(request);
      socket.send(reply, zmq::send_flags::none);
    } catch (zmq::error_t) {
      std::cout << "Loop aborted." << std::endl;
    }
  }
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
