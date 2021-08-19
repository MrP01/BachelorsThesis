import SEAL from "node-seal/throws_wasm_node_umd";

class BaseCommunicator {
  async init() {}
  async classify() {}
}

export class PlainCommunicator extends BaseCommunicator {
  async classify(flatImageArray) {
    const response = await fetch("http://localhost:8000/api/classify/plain/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: Array.from(flatImageArray),
      }),
    });
    return await response.json();
  }
}

export class SEALCommunicator extends BaseCommunicator {
  constructor() {
    super();
    this.seal = null;
    this.context = null;
  }

  async init() {
    // Wait for the library to initialize
    this.seal = await SEAL();
    const schemeType = this.seal.SchemeType.ckks;
    // const securityLevel = this.seal.SecurityLevel.tc128;
    const securityLevel = this.seal.SecurityLevel.none;
    const polyModulusDegree = 4096;
    const bitSizes = [50, 20, 50];

    const params = this.seal.EncryptionParameters(schemeType);
    params.setPolyModulusDegree(polyModulusDegree);

    // Create a suitable set of CoeffModulus primes
    params.setCoeffModulus(this.seal.CoeffModulus.Create(polyModulusDegree, Int32Array.from(bitSizes)));

    // Create a new Context
    this.context = this.seal.Context(
      params, // Encryption Parameters
      true, // ExpandModChain
      securityLevel // Enforce a security level
    );

    if (!this.context.parametersSet()) {
      throw new Error("Could not set the parameters in the given context. Please try different encryption parameters.");
    }

    // for debugging:
    window._seal = this.seal;
    window._context = this.context;
  }

  cool() {
    // Create a new KeyGenerator (creates a new keypair internally)
    const keyGenerator = this.seal.KeyGenerator(this.context);
    const secretKey = keyGenerator.secretKey();
    const publicKey = keyGenerator.createPublicKey();
    const relinKey = keyGenerator.createRelinKeys();
    const galoisKey = keyGenerator.createGaloisKeys(Int32Array.from([])); // Generating Galois keys takes a while compared to the others

    // Saving a key to a string is the same for each type of key
    // const secretBase64Key = secretKey.save()
    // const publicBase64Key = publicKey.save()
    const relinBase64Key = relinKey.save();
    const galoisBase64Key = galoisKey.save(); // saving Galois keys can take an even longer time and the output is **very** large.

    console.log("SEAL initialized.");

    const encoder = this.seal.CKKSEncoder(this.context);
    const encryptor = this.seal.Encryptor(this.context, publicKey);
    const scale = Math.pow(2, 20);
    var plaintext = encoder.encode(Float64Array.from([1, 2, 3]), scale);
    var ciphertext = encryptor.encrypt(plaintext);

    // console.log("Secret key", secretKey);
    // console.log("Public key", publicKey);
    // console.log("Relin key", relinBase64Key);
    // console.log("Galois key", galoisBase64Key);
  }
}
