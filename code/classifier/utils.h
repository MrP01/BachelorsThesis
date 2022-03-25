#include <seal/ciphertext.h>
#include <string>

inline void printCiphertextInternals(
    std::string name, seal::Ciphertext &x, seal::SEALContext *context, bool exact_scale = false) {
  std::cout << "> " << name << " scale: ";
  if (exact_scale)
    std::cout << std::fixed << x.scale();
  else
    std::cout << log2(x.scale()) << " bits";
  std::cout << "; parms chain index: " << context->get_context_data(x.parms_id())->chain_index()
            << "; size: " << x.size() << std::endl;
}
