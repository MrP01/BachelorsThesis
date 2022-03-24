#include "Network.h"

#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xreducer.hpp>
#include <xtensor/xsort.hpp>

Network::Network() {}

void Network::init() {
  seal::EncryptionParameters params(seal::scheme_type::ckks);
  size_t poly_modulus_degree = 4096; // same as for node-seal
  params.set_poly_modulus_degree(poly_modulus_degree);
  std::vector<int> bit_sizes = {60, 40, 40, 40, 40, 60};
  std::cout << "Product(bit_sizes) = " << xt::prod<int>(xt::adapt(bit_sizes, {bit_sizes.size()}))() << std::endl;
  params.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, bit_sizes));
  context = new seal::SEALContext(params, true, seal::sec_level_type::none);
}

void Network::addLayer(Layer *layer) {
  layers.push_back(layer);
  layer->parent = this;
}

Vector Network::predict(Vector input) {
  for (Layer *layer : layers) {
    input = layer->feedforward(input);
  }
  return input;
}

seal::Ciphertext Network::predictEncrypted(seal::Ciphertext &ciphertext, seal::RelinKeys &relinKeys,
                                           seal::GaloisKeys &galoisKeys) {
  seal::Evaluator evaluator(*context);
  seal::CKKSEncoder encoder(*context);

  int index = 0;
  for (Layer *layer : layers) {
    std::cout << "Feeding ciphertext forward through layer " << index++ << std::endl;
    layer->feedforwardEncrypted(ciphertext, galoisKeys, relinKeys, encoder, evaluator);
  }

  return ciphertext;
};

int Network::interpretResult(Vector result) { return xt::argmax(result)(); };

Vector Network::interpretResultProbabilities(Vector result) {
  Vector y = xt::exp(result);
  return y / xt::sum(y);
};
