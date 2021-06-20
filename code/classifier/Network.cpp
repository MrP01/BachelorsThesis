#include "Network.h"

#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>

Network::Network() {}

void Network::addLayer(Layer *layer) { layers.push_back(layer); }

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
