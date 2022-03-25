#include <plog/Log.h>
#include <seal/ciphertext.h>
#include <string>

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
