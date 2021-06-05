#include "Network.h"

#include <xtensor/xrandom.hpp>

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
