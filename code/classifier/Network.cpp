#include <xtensor/xrandom.hpp>
#include "Network.h"

Network::Network() {

}

void Network::addLayer(Layer* layer) {
  layers.push_back(layer);
}

void Network::addLayer(int neuronsIn, int neuronsOut) {
  auto layer = new Layer(
      xt::random::randn<double>({neuronsOut, neuronsIn}),
      xt::random::randn<double>({neuronsOut})
  );
  addLayer(layer);
}
