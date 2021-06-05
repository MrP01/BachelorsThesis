#include "vector"
#include "Layer.h"

class Network {
  private:
    std::vector<Layer*> layers;

  public:
    Network();
    void addLayer(Layer* layer);
    void addLayer(int neuronsIn, int neuronsOut);
    Vector predict(Vector input);
    int interpret_result(Vector result);
    Vector interpret_result_probabilities(Vector result);
};
