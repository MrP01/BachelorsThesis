#include <iostream>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>

#include "Network.h"

int main() {
  std::cout << "--- MNIST Neural Network Predictor ---" << std::endl;
  auto neuralNet = new Network();
  //  neuralNet->addLayer(784, 128);
  //  neuralNet->addLayer(128, 10);
  auto w1 = xt::load_npy<float>("data/models/simple/w1.npy");
  auto b1 = xt::load_npy<float>("data/models/simple/b1.npy");
  auto w2 = xt::load_npy<float>("data/models/simple/w2.npy");
  auto b2 = xt::load_npy<float>("data/models/simple/b2.npy");
  neuralNet->addLayer(new Layer(w1, b1));
  neuralNet->addLayer(new Layer(w2, b2));

  // auto x_train = xt::load_npy<float>("data/mnist/x-train.npy");
  // auto y_train = xt::load_npy<int>("data/mnist/y-train.npy");
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  auto y_test = xt::load_npy<int>("data/mnist/y-test.npy");
  int N = x_test.shape()[0];
  // x_test.reshape({N, 784});

  int correct = 0, i = 0;
  for (auto iter = xt::axis_begin(x_test, 0); iter != xt::axis_end(x_test, 0); iter++) {
    Vector x = *iter;
    x.reshape({784});
    Vector result = neuralNet->predict(x);
    int prediction = xt::argmax(result)();
    // std::cout << prediction << " | " << y_test[i] << std::endl;
    if (i % 50 == 0)
      std::cout << i << " / " << N << "\r" << std::flush;
    if (prediction == y_test[i])
      correct++;
    i++;
  }
  std::cout << std::endl;
  double accuracy = (double)correct / (double)N;
  std::cout << "Correct: " << correct << " out of " << N << std::endl;
  std::cout << "Model accuracy: " << accuracy << std::endl;
  return 0;
}
