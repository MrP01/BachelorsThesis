#include <xtensor/xmath.hpp>
#include "Layer.h"

Layer::Layer(Matrix weights, Vector biases) : weights(weights), biases(biases) {
  std::cout << "Hola from Layer" << std::endl;
}

Vector Layer::feedforward(Vector x) {
  return activation(weights * x + biases);
}

Vector Layer::activation(Vector x) {
  return 1 / (1 + xt::exp(-x));
}

Vector Layer::activationPrime(Vector x) {
  return activation(x) * (1 - activation(x));
}
