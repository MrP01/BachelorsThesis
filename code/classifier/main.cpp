#include <iostream>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>

#include "Network.h"

int main() {
  std::cout << "Hello, World!" << std::endl;
  auto neuralNet = new Network();
  //  neuralNet->addLayer(784, 128);
  //  neuralNet->addLayer(128, 10);
  auto w1 = xt::load_npy<float>("data/models/simple/w1.npy");
  auto b1 = xt::load_npy<float>("data/models/simple/b1.npy");
  auto w2 = xt::load_npy<float>("data/models/simple/w2.npy");
  auto b2 = xt::load_npy<float>("data/models/simple/b2.npy");
  neuralNet->addLayer(new Layer(w1, b1));
  neuralNet->addLayer(new Layer(w2, b2));

  auto x_train = xt::load_npy<float>("data/mnist/x-train.npy");
  auto y_train = xt::load_npy<int>("data/mnist/y-train.npy");
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  auto y_test = xt::load_npy<int>("data/mnist/y-test.npy");
  return 0;
}
