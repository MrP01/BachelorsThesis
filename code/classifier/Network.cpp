#include "Network.h"
#include "xtensor/xrandom.hpp"

Network::Network() {

}

void Network::addLayer(int neuronsIn, int neuronsOut) {
  auto layer = new Layer(
      xt::random::randn<double>({neuronsOut, neuronsIn}),
      xt::random::randn<double>({neuronsOut})
  );
  this->layers.push_back(layer);
}
