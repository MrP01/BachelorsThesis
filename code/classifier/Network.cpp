#include "Network.h"
#include <NTL/ZZ.h>
#include <NTL/ZZ_limbs.h>
#include <lodepng.h>
#include <plog/Log.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xreducer.hpp>
#include <xtensor/xsort.hpp>

Network::Network() {}

void Network::init() {
  parameters = new seal::EncryptionParameters(seal::scheme_type::ckks);
  size_t poly_modulus_degree = POLY_MOD_DEGREE;
  parameters->set_poly_modulus_degree(poly_modulus_degree);
  std::vector<int> bit_sizes;
  bit_sizes.push_back(COEFF_MODULUS_START_BITS);
  for (size_t i = 0; i < 5; i++)
    bit_sizes.push_back(COEFF_MODULUS_MIDDLE_BITS);
  bit_sizes.push_back(*std::max_element(bit_sizes.begin(), bit_sizes.end()));

  PLOG(plog::debug) << "PolyModDegree: " << poly_modulus_degree << " so we need " << 2 * log2(poly_modulus_degree) - 1
                    << " Galois keys.";

  PLOG(plog::debug) << "sum(bit_sizes) = " << xt::sum(xt::adapt(bit_sizes, {bit_sizes.size()}))();
  parameters->set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, bit_sizes));
  std::vector<double> log_coeff_moduli;
  for (auto &&modulus : parameters->coeff_modulus())
    log_coeff_moduli.push_back(log2(modulus.value()));
  PLOG(plog::debug) << "log2(product(moduli)) = " << xt::sum(xt::adapt(log_coeff_moduli, {log_coeff_moduli.size()}))();
  context = new seal::SEALContext(*parameters, true, SECURITY_LEVEL);
}

void Network::loadDefaultModel() {
  auto w1 = xt::load_npy<float>("data/models/simple/w1.npy");
  auto b1 = xt::load_npy<float>("data/models/simple/b1.npy");
  auto w2 = xt::load_npy<float>("data/models/simple/w2.npy");
  auto b2 = xt::load_npy<float>("data/models/simple/b2.npy");
  addLayer(new DenseLayer(w1, b1));
  addLayer(new ActivationLayer());
  addLayer(new DenseLayer(w2, b2));

  seal::Evaluator evaluator(*context);
  seal::CKKSEncoder encoder(*context);
  layers[0]->prepare(encoder, evaluator, context->first_parms_id(), SCALE); // chain index 5
  layers[2]->prepare(encoder, evaluator, context->last_context_data().get()->prev_context_data()->parms_id(), SCALE);
}

void Network::addLayer(Layer *layer) {
  layers.push_back(layer);
  layer->parent = this;
}

Vector Network::predict(Vector input) {
  size_t index = 0;
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

  size_t index = 0;
  for (Layer *layer : layers) {
    PLOG(plog::debug) << "Feeding ciphertext through layer " << index++;
    layer->feedforwardEncrypted(ciphertext, galoisKeys, relinKeys, encoder, evaluator);
  }

  return ciphertext;
}

Digit Network::interpretResult(Vector result) { return xt::argmax(result)(); }

Vector Network::interpretResultProbabilities(Vector result) {
  Vector y = xt::exp(result);
  return y / xt::sum(y);
}

xt::xarray<uint8_t> Network::interpretCiphertextAsPixels(seal::Ciphertext &ciphertext) {
  size_t N = ciphertext.poly_modulus_degree(); // obviously the number of coefficients in one polynomial component
  std::vector<seal::Modulus> rns_moduli = context->get_context_data(ciphertext.parms_id())->parms().coeff_modulus();
  NTL::ZZ full_modulus(1);
  for (size_t i = 0; i < rns_moduli.size(); i++)
    full_modulus *= rns_moduli[i].value();
  PLOG(plog::debug) << "Our Full Coeff Modulus q = " << full_modulus;

  seal::util::PolyIter polyIter(ciphertext);
  seal::util::RNSIter rnsIter(polyIter[0]); // iterator over polynomial c0 which we are interested in

  // because of unimplemented lambda function features in g++, we use references instead of the actual NTL::ZZ objects
  std::vector<NTL::ZZ *> a, p; // p is the product of the moduli ==> q
  size_t rns_component_index = 0;
  SEAL_ITERATE(rnsIter, rns_moduli.size(), [&](auto RNS) {
    PLOG(plog::debug) << "Evaluating RNS Component of the first polynomial";
    seal::util::CoeffIter coeffIter(RNS);
    NTL::ZZ P(rns_moduli[rns_component_index].value());

    size_t coeff_index = 0;
    SEAL_ITERATE(coeffIter, N, [&](auto Coeff) {
      if (rns_component_index == 0) {
        a.push_back(new NTL::ZZ(Coeff));
        p.push_back(new NTL::ZZ(rns_moduli[0].value()));
      } else
        NTL::CRT(*a[coeff_index], *p[coeff_index], NTL::ZZ(Coeff), P);
      coeff_index++;
    });
    rns_component_index++;
  });
  PLOG(plog::debug) << "a: " << *a[0] << ", p: " << *p[0];
  PLOG(plog::debug) << "a: " << *a[1] << ", p: " << *p[1];

  size_t width = 1 << (size_t)(log2(N) / 2 + 1), height = N / width;
  assert(width * height == N);
  xt::xarray<uint8_t> image = xt::zeros<uint8_t>({width, height});
  size_t n = 0;
  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < height; j++) {
      // PLOG(plog::debug) << "a: " << *a[i] << ", p: " << *p[i];
      NTL::ZZ pixel = *(a[n]) * 128 / full_modulus + 128;
      image(i, j) = NTL::ZZ_limbs_get(pixel)[0];
      n++;
    }
  }
  return image;
}

void Network::saveXArrayToPNG(std::string filename, xt::xarray<uint8_t> image) {
  size_t width = image.shape(0);
  size_t height = image.shape(1);
  std::vector<unsigned char> pixels;
  pixels.resize(width * height * 4);
  for (unsigned y = 0; y < height; y++)
    for (unsigned x = 0; x < width; x++) {
      auto brightness = image(x, y);
      pixels[4 * width * y + 4 * x + 0] = brightness;
      pixels[4 * width * y + 4 * x + 1] = brightness;
      pixels[4 * width * y + 4 * x + 2] = brightness;
      pixels[4 * width * y + 4 * x + 3] = 255;
    }
  lodepng::encode(filename, pixels, width, height);
}
