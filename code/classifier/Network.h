#include "vector"
#include "Layer.h"

class Network {
  private:
    std::vector<Layer*> layers;

  public:
    seal::SEALContext* context = nullptr;

  public:
    Network();
    void init();
    void addLayer(Layer* layer);
    void addLayer(int neuronsIn, int neuronsOut);
    Vector predict(Vector input);
    Vector predictEncrypted();
    int interpretResult(Vector result);
    Vector interpretResultProbabilities(Vector result);
};
