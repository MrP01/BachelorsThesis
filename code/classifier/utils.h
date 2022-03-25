#include <plog/Log.h>
#include <seal/ciphertext.h>
#include <string>

inline void printCiphertextInternals(
    std::string name, seal::Ciphertext &x, seal::SEALContext *context, bool exact_scale = false) {
  PLOG(plog::debug) << "> " << name << " scale: ";
  if (exact_scale)
    PLOG(plog::debug) << std::fixed << x.scale();
  else
    PLOG(plog::debug) << log2(x.scale()) << " bits";
  PLOG(plog::debug) << "; parms chain index: " << context->get_context_data(x.parms_id())->chain_index()
                    << "; size: " << x.size();
}
