#include "Network.h"

#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>

Network::Network() {}

void Network::init() {
  seal::EncryptionParameters params(seal::scheme_type::ckks);
  size_t poly_modulus_degree = 4096; // same as for node-seal
  params.set_poly_modulus_degree(poly_modulus_degree);
  params.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, {50, 20, 50}));
  context = new seal::SEALContext(params, true, seal::sec_level_type::none);
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

seal::Ciphertext Network::predictEncrypted(seal::Ciphertext &ciphertext, seal::RelinKeys &relinKeys, seal::GaloisKeys &galoisKeys) {
  seal::Evaluator evaluator(*context);
  seal::CKKSEncoder encoder(*context);

  for (Layer *layer : layers) {
    std::cout << "Feeding ciphertext forward through layer" << std::endl;
    layer->feedforwardEncrypted(ciphertext, galoisKeys, relinKeys, encoder, evaluator);
  }

  return ciphertext;
};

int Network::interpretResult(Vector result) { return xt::argmax(result)(); };

Vector Network::interpretResultProbabilities(Vector result) {
  Vector y = xt::exp(result / (xt::amax(xt::abs(result)) / 8));
  return y / xt::sum(y);
};
