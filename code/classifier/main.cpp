#include <iostream>
#include <xtensor/xnpy.hpp>
#include <xtensor/xio.hpp>
#include "Network.h"

int main() {
  std::cout << "Hello, World!" << std::endl;
  auto neuralNet = new Network();
  neuralNet->addLayer(784, 20);
  neuralNet->addLayer(20, 20);
  neuralNet->addLayer(20, 10);

  auto x_train = xt::load_npy<float>("data/mnist/x-train.npy");
  auto y_train = xt::load_npy<int>("data/mnist/y-train.npy");
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  auto y_test = xt::load_npy<int>("data/mnist/y-test.npy");
  std::cout << x_test << std::endl;
  std::cout << y_test << std::endl;
  return 0;
}
