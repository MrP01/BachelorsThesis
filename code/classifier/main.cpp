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

zmq::message_t handle_prediction_request(zmq::message_t &request) {
  xt::xarray<double> input;
  try {
    nlohmann::json input_json = nlohmann::json::parse(request.to_string());
    xt::from_json(input_json, input);
  } catch (nlohmann::json::parse_error &ex) {
    std::cerr << "parse error at byte " << ex.byte << std::endl;
    return zmq::message_t("parse_error", 12);
  }
  assert(input.shape()[0] == 28);
  assert(input.shape()[1] == 28);
  assert(input.dimension() == 2);
  input.reshape({784});
  std::cout << "Incoming data is valid, predicting ..." << std::endl;
  Vector result = neuralNet->predict(input);
  nlohmann::json response = {
      {"prediction", neuralNet->interpret_result(result)},
      {"probabilites", neuralNet->interpret_result_probabilities(result)},
  };
  std::cout << "... replying with " << response << std::endl;
  std::string serialized = response.dump();
  return zmq::message_t(serialized.c_str(), serialized.length());
}

void run_server() {
  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_REP);
  socket.bind("tcp://*:5555");

  while (true) {
    zmq::message_t request;
    auto sent = socket.recv(request);
    std::cout << "Handling request ..." << std::endl;
    zmq::message_t reply = handle_prediction_request(request);
    socket.send(reply, zmq::send_flags::none);
  }
}

double evaluate_network_on_test_data() {
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  auto y_test = xt::load_npy<int>("data/mnist/y-test.npy");
  int N = x_test.shape()[0];
  // x_test.reshape({N, 784});

  int correct = 0, i = 0;
  for (auto iter = xt::axis_begin(x_test, 0); iter != xt::axis_end(x_test, 0); iter++) {
    Vector x = *iter;
    x.reshape({784});
    Vector result = neuralNet->predict(x);
    int prediction = neuralNet->interpret_result(result);
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

int main() {
  std::cout << "--- MNIST Neural Network Predictor ---" << std::endl;

  //  neuralNet->addLayer(784, 128);
  //  neuralNet->addLayer(128, 10);
  auto w1 = xt::load_npy<float>("data/models/simple/w1.npy");
  auto b1 = xt::load_npy<float>("data/models/simple/b1.npy");
  auto w2 = xt::load_npy<float>("data/models/simple/w2.npy");
  auto b2 = xt::load_npy<float>("data/models/simple/b2.npy");
  neuralNet->addLayer(new Layer(w1, b1));
  neuralNet->addLayer(new Layer(w2, b2));

  // evaluate_network_on_test_data(neuralNet);
  run_server();
  return 0;
}
