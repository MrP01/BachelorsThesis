#include "Network.h"

#include <plog/Log.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xreducer.hpp>
#include <xtensor/xsort.hpp>

Network::Network() {}

void Network::init() {
  seal::EncryptionParameters params(seal::scheme_type::ckks);
  size_t poly_modulus_degree = 16384; // same as for node-seal
  params.set_poly_modulus_degree(poly_modulus_degree);
  std::vector<int> bit_sizes = {60, 40, 40, 40, 40, 40, 60};
  PLOG(plog::debug) << "sum(bit_sizes) = " << xt::sum(xt::adapt(bit_sizes, {bit_sizes.size()}))();
  params.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, bit_sizes));
  std::vector<double> log_coeff_moduli;
  for (auto &&modulus : params.coeff_modulus())
    log_coeff_moduli.push_back(log2(modulus.value()));
  PLOG(plog::debug) << "log2(product(moduli)) = " << xt::sum(xt::adapt(log_coeff_moduli, {log_coeff_moduli.size()}))();
  context = new seal::SEALContext(params, true, seal::sec_level_type::tc128);
}

void Network::loadDefaultModel() {
  auto w1 = xt::load_npy<float>("data/models/simple/w1.npy");
  auto b1 = xt::load_npy<float>("data/models/simple/b1.npy");
  auto w2 = xt::load_npy<float>("data/models/simple/w2.npy");
  auto b2 = xt::load_npy<float>("data/models/simple/b2.npy");
  addLayer(new DenseLayer(w1, b1));
  addLayer(new ActivationLayer());
  addLayer(new DenseLayer(w2, b2));
}

void Network::addLayer(Layer *layer) {
  layers.push_back(layer);
  layer->parent = this;
}

Vector Network::predict(Vector input) {
  int index = 0;
  for (Layer *layer : layers) {
    PLOG(plog::debug) << "Feeding plain data through layer " << index++;
    input = layer->feedforward(input);
  }
  return input;
}

seal::Ciphertext Network::predictEncrypted(
    seal::Ciphertext &ciphertext, seal::RelinKeys &relinKeys, seal::GaloisKeys &galoisKeys) {
  seal::Evaluator evaluator(*context);
  seal::CKKSEncoder encoder(*context);

  int index = 0;
  for (Layer *layer : layers) {
    PLOG(plog::debug) << "Feeding ciphertext through layer " << index++;
    layer->feedforwardEncrypted(ciphertext, galoisKeys, relinKeys, encoder, evaluator);
  }

  return ciphertext;
};

int Network::interpretResult(Vector result) { return xt::argmax(result)(); };

Vector Network::interpretResultProbabilities(Vector result) {
  Vector y = xt::exp(result);
  return y / xt::sum(y);
};
