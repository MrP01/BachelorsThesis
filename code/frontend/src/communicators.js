import SEAL from "node-seal/throws_wasm_web_es";
import * as msgpack from "@msgpack/msgpack";

const API_URL = "/api";

class BaseCommunicator {
  static _singleton = null;

  async init() {
    return this;
  }

  static async instance() {
    if (this._singleton === null) {
      this._singleton = new this();
      return this._singleton.init();
    }
    return new Promise((resolve, reject) => {
      resolve(this._singleton);
    });
  }

  async _makeApiRequest(method, path, body) {
    if (method === "POST") {
      body = msgpack.encode(body);
      console.log("Sending msgpack-encoded data of", body.byteLength, "bytes");
    } else body = null;
    const response = await fetch(API_URL + path, {
      method: method,
      headers: { "Content-Type": "application/x-msgpack" },
      body: body,
    });
    if (!response.ok) throw new Error("[API] request completion failed.");
    return await msgpack.decodeAsync(response.body);
  }

  async classify() {}
  isSecure() {}
  delete() {}
}

export class PlainCommunicator extends BaseCommunicator {
  async classify(flatImageArray) {
    return await this._makeApiRequest("POST", "/classify/plain/", {
      image: Array.from(flatImageArray),
    });
  }

  isSecure() {
    return false;
  }
}

export class SEALCommunicator extends BaseCommunicator {
  constructor() {
    super();
    this.seal = null;
    this.context = null;
    this.scale = null;
  }

  async _initContext() {
    const paramsResponse = await this._makeApiRequest("GET", "/parameters/", {});
    this.scale = paramsResponse.scale;
    this.seal = await SEAL();
    const params = this.seal.EncryptionParameters(this.seal.SchemeType.ckks);
    params.loadArray(paramsResponse.parms);

    const securityLevel = {
      none: this.seal.SecurityLevel.none,
      128: this.seal.SecurityLevel.tc128,
      192: this.seal.SecurityLevel.tc192,
      256: this.seal.SecurityLevel.tc256,
    }[paramsResponse.security_level];
    this.context = this.seal.Context(params, true, securityLevel);

    if (!this.context.parametersSet()) {
      throw new Error("[SEAL] Could not set the parameters in the given context.");
    }

    // for debugging:
    window._seal = this.seal;
    window._context = this.context;
    console.log("[SEAL] initialized context.");
    return this.context;
  }

  _createKeys() {
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
    return this;
  }

  static argmax(array) {
    return [].reduce.call(array, (m, c, i, arr) => (c > arr[m] ? i : m), 0);
  }

  static softmax(array) {
    let exponentiated = array.map((x) => Math.exp(x));
    let sum = exponentiated.reduce((a, b) => a + b);
    return exponentiated.map((y) => y / sum);
  }

  async classify(flatImageArray) {
    const encoder = this.seal.CKKSEncoder(this.context);
    var plaintext = encoder.encode(Float64Array.from(flatImageArray), this.scale);

    // in order to use encryptSymmetric (Seal serializables in seed mode), the encryptor also requires the secret key
    const encryptor = this.seal.Encryptor(this.context, this._publicKey, this._secretKey);
    var ciphertext = encryptor.encryptSymmetricSerializable(plaintext);

    // saving Galois keys can take an even longer time and the output is **very** large.
    var response = await this._makeApiRequest("POST", "/classify/encrypted/", {
      ciphertext: ciphertext.saveArray(this.seal.ComprModeType.zstd),
      relinKeys: this._relinKeys.saveArray(this.seal.ComprModeType.zstd),
      galoisKeys: this._galoisKeys.saveArray(this.seal.ComprModeType.zstd),
    });

    console.log("Got encrypted response");
    var resultCiphertext = this.seal.CipherText();
    resultCiphertext.loadArray(this.context, response["result"]);
    const decryptor = this.seal.Decryptor(this.context, this._secretKey);
    var resultPlaintext = this.seal.PlainText();
    decryptor.decrypt(resultCiphertext, resultPlaintext);
    var result = encoder.decode(resultPlaintext);
    window._result = result;
    result = result.slice(0, 10);

    console.log("Decoded result");
    return {
      prediction: SEALCommunicator.argmax(result),
      probabilities: Array.from(SEALCommunicator.softmax(result)),
    };
  }

  delete() {
    this.context.delete();
    this._secretKey.delete();
    this._publicKey.delete();
    this._relinKeys.delete();
    this._galoisKeys.delete();
  }

  isSecure() {
    return true;
  }
}
