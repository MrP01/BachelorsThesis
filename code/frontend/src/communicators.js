import SEAL from "node-seal/throws_wasm_web_es";
import * as msgpack from "@msgpack/msgpack";

const API_URL = "/api";

class BaseCommunicator {
  static _singleton = null;

  async init() {}
  static instance() {
    if (this._singleton === null) {
      this._singleton = new this();
      this._singleton.init().then(() => console.log("Communicator initialized."));
    }
    return this._singleton;
  }
  async classify() {}
  async _makeApiRequest(path, body) {
    // console.log("sending", body);
    const response = await fetch(API_URL + path, {
      method: "POST",
      headers: { "Content-Type": "application/x-msgpack" },
      body: msgpack.encode(body),
    });
    if (!response.ok) {
      throw new Error("[API] request completion failed.");
    }
    return await msgpack.decodeAsync(response.body);
  }
  delete() {}
}

export class PlainCommunicator extends BaseCommunicator {
  async classify(flatImageArray) {
    return await this._makeApiRequest("/classify/plain/", {
      image: Array.from(flatImageArray),
    });
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
    const bitSizes = [60, 40, 40, 40, 40, 40, 60];

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
      throw new Error(
        "[SEAL] Could not set the parameters in the given context. Please try different encryption parameters."
      );
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
    this._relinKeys = keyGenerator.createRelinKeys();
    this._galoisKeys = keyGenerator.createGaloisKeys(Int32Array.from([])); // Generating Galois keys takes a while compared to the others
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
    var response = await this._makeApiRequest("/classify/encrypted/", {
      ciphertext: ciphertext.saveArray(this.seal.ComprModeType.zstd),
      relinKeys: this._relinKeys.saveArray(this.seal.ComprModeType.zstd),
      galoisKeys: this._galoisKeys.saveArray(this.seal.ComprModeType.zstd), // saving Galois keys can take an even longer time and the output is **very** large.
    });
    var resultCiphertext = this.seal.CipherText();
    resultCiphertext.loadArray(this.context, response["result"]);
    const decryptor = this.seal.Decryptor(this.context, this._secretKey);
    var resultPlaintext = this.seal.PlainText();
    decryptor.decrypt(resultCiphertext, resultPlaintext);
    var result = encoder.decode(resultPlaintext); // TODO: apply softmax and get prediction
    return result;
  }

  delete() {
    this.context.delete();
    this._secretKey.delete();
    this._publicKey.delete();
    this._relinKeys.delete();
    this._galoisKeys.delete();
  }
}
