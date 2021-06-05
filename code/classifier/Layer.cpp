#include "Layer.h"
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xmath.hpp>

Layer::Layer(Matrix weights, Vector biases) : weights(weights), biases(biases) {}

Vector Layer::feedforward(Vector x) {
  // std::cout << x.shape()[0] << " - " << weights.shape()[0] << weights.shape()[1] << std::endl;
  // std::cout << xt::view(x) << std::endl;
  Vector dot = xt::zeros<double>({weights.shape()[1]});
  int i = 0;
  for (auto iter = xt::axis_begin(weights, 1); iter != xt::axis_end(weights, 1); iter++) {
    dot[i] = xt::sum((*iter) * x)();
    i++;
  }
  // std::copy(dot.begin(), dot.end(), std::ostream_iterator<float>(std::cout, ", "));
  return activation(dot + biases);
}

Vector Layer::activation(Vector x) { return 0.54738 + 0.59579 * x + 0.090189 * xt::pow(x, 2) - 0.006137 * xt::pow(x, 3); }
