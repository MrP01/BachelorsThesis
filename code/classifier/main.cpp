#include <backward.hpp>
#include <csignal>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/FuncMessageFormatter.h>
#include <plog/Log.h>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xjson.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>

#include "Network.h"

Network neuralNet;
bool quit = false;

namespace backward {
backward::SignalHandling _signalHandler;
}

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
      // {"probabilities", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, // prediction and probabilities should be calculated
      // client-side
  };
}

nlohmann::json handleGetTestData(nlohmann::json request) {
  auto x_test = xt::load_npy<float>("data/mnist/x-test.npy");
  typedef std::vector<std::vector<uint8_t>> Image;
  std::vector<Image> images;
  for (auto &&index : request["indices"]) {
    auto view = xt::view(x_test, index, xt::all());
    images.push_back(Image(view.begin(), view.end()));
  }
  return nlohmann::json(images);
}

auto msgpackRequestHandler(nlohmann::json (*handler)(nlohmann::json)) {
  return
      [=](const httplib::Request &request, httplib::Response &response, const httplib::ContentReader &contentReader) {
        std::string request_body;
        contentReader([&](const char *data, size_t data_length) {
          request_body.append(data, data_length);
          return true;
        });
        nlohmann::json request_json = nlohmann::json::from_msgpack(request_body);
        nlohmann::json response_json = handler(request_json);
        std::vector<uint8_t> serialized = nlohmann::json::to_msgpack(response_json);
        response.set_content(std::string(serialized.begin(), serialized.end()), "application/x-msgpack");
      };
}

void runServer() {
  httplib::Server server;
  server.Post("/api/classify/plain/", msgpackRequestHandler(handlePlainPredictionRequest));
  server.Post("/api/classify/encrypted/", msgpackRequestHandler(handleEncryptedPredictionRequest));

  server.set_exception_handler([](const httplib::Request &req, httplib::Response &res, std::exception &exception) {
    PLOG(plog::debug) << "Exception caught: " << exception.what();
    backward::StackTrace trace;
    trace.load_here();
    backward::Printer printer;
    printer.print(trace);
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
  quit = true;
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
