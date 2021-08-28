#include "Layer.h"
#include "vector"

class Network {
  private:
    std::vector<Layer *> layers;

  public:
    seal::SEALContext *context = nullptr;

  public:
    Network();
    void init();
    void addLayer(Layer *layer);
    void addLayer(int neuronsIn, int neuronsOut);
    Vector predict(Vector input);
    seal::Ciphertext predictEncrypted(seal::Ciphertext &ciphertext, seal::RelinKeys &relinKeys, seal::GaloisKeys &galoisKeys);
    int interpretResult(Vector result);
    Vector interpretResultProbabilities(Vector result);
};
