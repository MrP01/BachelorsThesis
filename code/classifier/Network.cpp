#include "Network.h"

#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>

Network::Network() {}

void Network::init() {
  seal::EncryptionParameters params(seal::scheme_type::ckks);
  size_t poly_modulus_degree = 4096; // same as for node-seal
  params.set_poly_modulus_degree(poly_modulus_degree);
  params.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, {50, 20, 50}));
  context = new seal::SEALContext(params);
}

void Network::addLayer(Layer *layer) {
  layers.push_back(layer);
  layer->parent = this;
}

void Network::addLayer(int neuronsIn, int neuronsOut) {
  auto layer = new Layer(xt::random::randn<double>({neuronsOut, neuronsIn}), xt::random::randn<double>({neuronsOut}));
  addLayer(layer);
}

Vector Network::predict(Vector input) {
  for (Layer *layer : layers) {
    input = layer->feedforward(input);
  }
  return input;
}

int Network::interpretResult(Vector result) { return xt::argmax(result)(); };

Vector Network::interpretResultProbabilities(Vector result) {
  Vector y = xt::exp(result / (xt::amax(xt::abs(result)) / 8));
  return y / xt::sum(y);
};
