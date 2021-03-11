#include <iostream>
#include "Network.h"

int main() {
  std::cout << "Hello, World!" << std::endl;
  auto neuralNet = new Network();
  neuralNet->addLayer(784, 20);
  neuralNet->addLayer(20, 20);
  neuralNet->addLayer(20, 10);
  return 0;
}
