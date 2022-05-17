#include <csignal>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/FuncMessageFormatter.h>
#include <plog/Log.h>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xjson.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>

#include "Network.h"

Network neuralNet;
httplib::Server server;

nlohmann::json handlePlainPredictionRequest(nlohmann::json request) {
  xt::xarray<double> input;
  xt::from_json(request["image"], input);
  PLOG(plog::debug) << input;
  assert(input.dimension() == 1);
  assert(input.shape()[0] == 784);
  PLOG(plog::debug) << "Incoming data is valid, predicting ...";
  Vector result = neuralNet.predict(input);
  return nlohmann::json{
      {"prediction", neuralNet.interpretResult(result)},
      {"probabilities", neuralNet.interpretResultProbabilities(result)},
  };
}

nlohmann::json handleEncryptedPredictionRequest(nlohmann::json request) {
  PLOG(plog::debug) << "Incoming encrypted request";

  seal::RelinKeys relinKeys;
  seal::GaloisKeys galoisKeys;
  seal::Ciphertext ciphertext;

  nlohmann::json::binary_t binary = request["relinKeys"].get<nlohmann::json::binary_t>();
  PLOG(plog::debug) << "Decoded length: " << binary.size();
  PLOG(plog::debug) << "NeuralNet context poly mod degree: "
                    << neuralNet.context->key_context_data()->parms().poly_modulus_degree();
  std::stringstream dataStream = std::stringstream(std::string(binary.begin(), binary.end()));
  assert(seal::Serialization::compr_mode_default == seal::compr_mode_type::zstd);
  relinKeys.load(*neuralNet.context, dataStream);

  binary = request["galoisKeys"].get<nlohmann::json::binary_t>();
  dataStream = std::stringstream(std::string(binary.begin(), binary.end()));
  galoisKeys.load(*neuralNet.context, dataStream);

  binary = request["ciphertext"].get<nlohmann::json::binary_t>();
  dataStream = std::stringstream(std::string(binary.begin(), binary.end()));
  ciphertext.load(*neuralNet.context, dataStream);

  seal::Ciphertext result = neuralNet.predictEncrypted(ciphertext, relinKeys, galoisKeys);

  std::vector<uint8_t> byte_buffer(static_cast<size_t>(result.save_size()));
  result.save(reinterpret_cast<seal::seal_byte *>(byte_buffer.data()), byte_buffer.size());
  auto binaryResult = nlohmann::json::binary(byte_buffer);
  return nlohmann::json{
      {"result", binaryResult},
  };
}

void handleGetTestData(const httplib::Request &req, httplib::Response &response) {
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  size_t pos = 0;
  std::vector<int> indices;
  std::string indices_str = req.get_param_value("indices");
  while ((pos = indices_str.find("-")) != std::string::npos) {
    indices.push_back(std::atoi(indices_str.substr(0, pos).c_str()));
    indices_str.erase(0, pos + 1);
  }
  indices.push_back(std::atoi(indices_str.c_str()));
  nlohmann::json data;
  xt::to_json(data, xt::view(x_test, xt::keep(indices)));
  response.set_content(data.dump(), "application/json");
}

void runServer() {
  server.Post("/api/classify/plain/", msgpackRequestHandler(handlePlainPredictionRequest));
  server.Post("/api/classify/encrypted/", msgpackRequestHandler(handleEncryptedPredictionRequest));
  server.Get("/api/testdata/", handleGetTestData);

  server.set_exception_handler([](const httplib::Request &req, httplib::Response &res, std::exception &exception) {
    PLOG(plog::debug) << "Exception caught: " << exception.what();
    res.status = 500;
    res.set_content(exception.what(), "text/plain");
  });
  server.set_logger([](const httplib::Request &req, const httplib::Response &res) {
    // prints log after the response was sent
    PLOG(plog::info) << "[" << req.method << "] " << req.path << " " << res.status;
  });

  PLOG(plog::info) << "The server is running";
  server.listen("0.0.0.0", 8000);
}

void shutdown(int signum) {
  PLOG(plog::info) << "Shutdown...";
  server.stop();
}

int main() {
  static plog::ColorConsoleAppender<plog::FuncMessageFormatter> appender;
  plog::init(plog::debug, &appender);
  PLOG(plog::info) << "--- MNIST Neural Network Predictor ---";
  neuralNet.init();
  neuralNet.loadDefaultModel();
  signal(SIGTERM, shutdown);
  runServer();
  return 0;
}
