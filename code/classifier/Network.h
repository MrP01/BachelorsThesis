#include "Layer.h"
#include "vector"

class Network {
 public:
  std::vector<Layer *> layers;
  seal::SEALContext *context = nullptr;

 public:
  Network();
  void init();
  void loadDefaultModel();
  void addLayer(Layer *layer);
  Vector predict(Vector input);
  seal::Ciphertext predictEncrypted(
      seal::Ciphertext &ciphertext, seal::RelinKeys &relinKeys, seal::GaloisKeys &galoisKeys);
  int interpretResult(Vector result);
  Vector interpretResultProbabilities(Vector result);

  xt::xarray<uint8_t> interpretCiphertextAsPixels(seal::Ciphertext &ciphertext);
  static void saveXArrayToPNG(std::string filename, xt::xarray<uint8_t> image);
};
