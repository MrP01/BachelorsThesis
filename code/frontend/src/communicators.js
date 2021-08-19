import SEAL from "node-seal/throws_wasm_node_umd";

const API_URL = "http://localhost:8000";

class BaseCommunicator {
  async init() {}
  async classify() {}
  async _makeApiRequest(path, body) {
    const response = await fetch(API_URL + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body,
    });
    if (!response.ok) {
      throw new Error("[API] request completion failed.");
    }
    return await response.json();
  }
  delete() {}
}

export class PlainCommunicator extends BaseCommunicator {
  async classify(flatImageArray) {
    return await this._makeApiRequest(
      "/api/classify/plain/",
      JSON.stringify({
        image: Array.from(flatImageArray),
      })
    );
  }
}

export class SEALCommunicator extends BaseCommunicator {
  constructor() {
    super();
    this.seal = null;
    this.context = null;
  }

  async _initContext() {
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
      throw new Error("[SEAL] Could not set the parameters in the given context. Please try different encryption parameters.");
    }

    // for debugging:
    window._seal = this.seal;
    window._context = this.context;
    console.log("[SEAL] initialized context.");
    return this.context;
  }

  _createKeys() {
    // Create a new KeyGenerator (creates a new keypair internally)
    const keyGenerator = this.seal.KeyGenerator(this.context);
    this._secretKey = keyGenerator.secretKey();
    this._publicKey = keyGenerator.createPublicKey();
    this._relinKey = keyGenerator.createRelinKeys();
    this._galoisKey = keyGenerator.createGaloisKeys(Int32Array.from([])); // Generating Galois keys takes a while compared to the others
    console.log("[SEAL] keys created.");
    keyGenerator.delete();
  }

  async init() {
    await this._initContext();
    this._createKeys();
  }

  async classify(flatImageArray) {
    const encoder = this.seal.CKKSEncoder(this.context);
    const encryptor = this.seal.Encryptor(this.context, this._publicKey);
    const scale = Math.pow(2, 20);
    var plaintext = encoder.encode(Float64Array.from(flatImageArray), scale);
    var ciphertext = encryptor.encrypt(plaintext);
    return await this._makeApiRequest(
      "/api/classify/encrypted/",
      JSON.stringify({
        ciphertext: ciphertext.save(),
        relinKey: this._relinKey.save(),
        galoisKey: this._galoisKey.save(), // saving Galois keys can take an even longer time and the output is **very** large.
      })
    );
  }

  delete() {
    this.context.delete();
    this._secretKey.delete();
    this._publicKey.delete();
    this._relinKey.delete();
    this._galoisKey.delete();
  }
}
