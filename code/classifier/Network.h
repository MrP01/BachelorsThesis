#include "vector"
#include "Layer.h"

class Network {
  private:
    std::vector<Layer*> layers;

  public:
    Network();
    void addLayer(Layer* layer);
    void addLayer(int neuronsIn, int neuronsOut);
};
