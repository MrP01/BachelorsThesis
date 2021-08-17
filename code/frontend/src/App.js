import "@materializecss/materialize";
import "@materializecss/materialize/dist/css/materialize.css";
import { Button, Col, Icon, Navbar, Row } from "react-materialize";
import { ReactPainter } from "react-painter";
import "./App.css";
import SEAL from "node-seal/throws_wasm_node_umd";
import Pica from "pica";

const pica = new Pica();

function classify() {
  console.log("Classifying given input");
  const source = document.querySelector("canvas");
  const target = document.querySelector("#target-28x28");
  pica.resize(source, target, { alpha: true }).then((result) => {
    console.log("Resizing to 28x28 finished.");
    let ctx = result.getContext("2d");
    let alphaChannel = ctx
      .getImageData(0, 0, 28, 28)
      .data.filter((value, index) => index % 4 === 3); // alpha channel is every 4th element
    console.log(alphaChannel);
  });
}

(async () => {
  // Wait for the library to initialize
  const seal = await SEAL();
  const schemeType = seal.SchemeType.ckks;
  // const securityLevel = seal.SecurityLevel.tc128;
  const securityLevel = seal.SecurityLevel.none;
  const polyModulusDegree = 4096;
  const bitSizes = [50, 20, 50];

  const params = seal.EncryptionParameters(schemeType);
  params.setPolyModulusDegree(polyModulusDegree);

  // Create a suitable set of CoeffModulus primes
  params.setCoeffModulus(
    seal.CoeffModulus.Create(polyModulusDegree, Int32Array.from(bitSizes))
  );

  // Create a new Context
  const context = seal.Context(
    params, // Encryption Parameters
    true, // ExpandModChain
    securityLevel // Enforce a security level
  );
  window.seal = seal;
  window.context = context;

  if (!context.parametersSet()) {
    throw new Error(
      "Could not set the parameters in the given context. Please try different encryption parameters."
    );
  }

  // Create a new KeyGenerator (creates a new keypair internally)
  const keyGenerator = seal.KeyGenerator(context);

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
  console.log("Secret key", secretKey);
  console.log("Public key", publicKey);
  console.log("Relin key", relinBase64Key);
  console.log("Galois key", galoisBase64Key);
})();

function App() {
  return (
    <div>
      <Navbar
        alignLinks="right"
        brand={<span style={{ paddingLeft: "10px" }}>FHE Classifier</span>}
        id="mobile-nav"
        menuIcon={<Icon>menu</Icon>}
      ></Navbar>
      <Row>
        <h3 className={"center"}>Classify your Secret Data</h3>
        <p>
          Using Fully Homomorphic Encryption, directly from within the browser.
        </p>
        <Col m={6} s={12}>
          <ReactPainter
            width={300}
            height={300}
            initialColor={"cornflowerblue"}
            initialLineWidth={20}
            initialLineJoin={"miter"}
            lineCap={"round"}
            onSave={classify}
            render={({ triggerSave, canvas, setColor }) => (
              <div>
                <div className={"canvas-container center"}>{canvas}</div>
                <Button onClick={triggerSave}>Classify</Button>
              </div>
            )}
          />
        </Col>
        <Col m={6}>
          <canvas id="target-28x28" width={28} height={28}></canvas>
        </Col>
      </Row>
    </div>
  );
}

export default App;
