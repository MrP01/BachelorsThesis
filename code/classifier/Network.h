#include "Layer.h"
#include <vector>

#define POLY_MOD_DEGREE 8192
#define COEFF_MODULUS_START_BITS 34
#define COEFF_MODULUS_MIDDLE_BITS 25
// #define POLY_MOD_DEGREE 16384
// #define COEFF_MODULUS_START_BITS 60
// #define COEFF_MODULUS_MIDDLE_BITS 40
#define SECURITY_LEVEL seal::sec_level_type::tc128
#define SCALE pow(2.0, COEFF_MODULUS_MIDDLE_BITS)

class Network {
 public:
  std::vector<Layer *> layers;
  seal::EncryptionParameters *parameters;
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
