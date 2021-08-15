import '@materializecss/materialize';
import '@materializecss/materialize/dist/css/materialize.css';
import { Button, Col, Icon, Navbar, Row } from 'react-materialize';
import { ReactPainter } from 'react-painter';
import "./App.css";
import SEAL from 'node-seal/throws_wasm_node_umd';

function classify(blob) {
  console.log("Classifying given input", blob);
}

; (async () => {
  // Wait for the library to initialize
  const seal = await SEAL();
  const schemeType = seal.SchemeType.ckks;
  // const securityLevel = seal.SecurityLevel.tc128;
  const securityLevel = seal.SecurityLevel.none;
  const polyModulusDegree = 4096;
  const bitSizes = [50, 20, 50];

  const params = seal.EncryptionParameters(schemeType);
  params.setPolyModulusDegree(polyModulusDegree)

  // Create a suitable set of CoeffModulus primes
  params.setCoeffModulus(seal.CoeffModulus.Create(polyModulusDegree, Int32Array.from(bitSizes)));

  // Create a new Context
  const context = seal.Context(
    params, // Encryption Parameters
    true, // ExpandModChain
    securityLevel // Enforce a security level
  );
  window.seal = seal;
  window.context = context;

  if (!context.parametersSet()) {
    throw new Error('Could not set the parameters in the given context. Please try different encryption parameters.')
  }

  // Create a new KeyGenerator (creates a new keypair internally)
  const keyGenerator = seal.KeyGenerator(context)

  const secretKey = keyGenerator.secretKey()
  const publicKey = keyGenerator.createPublicKey()
  const relinKey = keyGenerator.createRelinKeys()
  // Generating Galois keys takes a while compared to the others
  const galoisKey = keyGenerator.createGaloisKeys(Int32Array.from([]));

  // Saving a key to a string is the same for each type of key
  // const secretBase64Key = secretKey.save()
  // const publicBase64Key = publicKey.save()
  const relinBase64Key = relinKey.save()
  // Please note saving Galois keys can take an even longer time and the output is **very** large.
  const galoisBase64Key = galoisKey.save();
  // const contextBase64 = context.save();
})()

function App() {
  return (
    <div>
      <Navbar
        alignLinks="right"
        brand={<span style={{ "paddingLeft": "10px" }}>FHE Classifier</span>}
        id="mobile-nav"
        menuIcon={<Icon>menu</Icon>}
      >
      </Navbar>
      <Row>
        <Col s={12}>
          <h3 className={"center"}>Classify your Secret Data</h3>
          <p>Using Fully Homomorphic Encryption, directly from within the browser.</p>
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
      </Row>
    </div>
  );
}

export default App;
