#include <httplib.h>
#include <nlohmann/json.hpp>
#include <plog/Log.h>
#include <seal/ciphertext.h>
#include <seal/ckks.h>
#include <seal/decryptor.h>
#include <seal/plaintext.h>
#include <string>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

inline void printCiphertextInternals(
    std::string name, seal::Ciphertext &x, seal::SEALContext *context, bool exact_scale = false) {
  std::stringstream message;
  message << "> " << name << " scale: ";
  if (exact_scale)
    message << std::fixed << x.scale();
  else
    message << log2(x.scale()) << " bits";
  message << "; parms chain index: " << context->get_context_data(x.parms_id())->chain_index()
          << "; size: " << x.size();
  PLOG(plog::debug) << message.str();
}

inline xt::xarray<double> getCiphertextValue(
    seal::Ciphertext &x, size_t n, seal::Decryptor *decryptor, seal::CKKSEncoder &encoder) {
  seal::Plaintext plain;
  std::vector<double> decoded_plain;

  decryptor->decrypt(x, plain);
  encoder.decode(plain, decoded_plain);
  xt::xarray<double> result = xt::adapt(decoded_plain, {n});
  // PLOG(plog::debug) << "      " << result;
  return result;
}

inline auto msgpackRequestHandler(nlohmann::json (*handler)(nlohmann::json)) {
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
