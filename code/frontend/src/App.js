import "@materializecss/materialize";
import "@materializecss/materialize/dist/css/materialize.css";
import { Button, Col, Icon, Navbar, Row, Container } from "react-materialize";
import { ReactPainter } from "react-painter";
import "./App.css";
import SEAL from "node-seal/throws_wasm_node_umd";
import Pica from "pica";
import React from "react";

const pica = new Pica();

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
    throw new Error("Could not set the parameters in the given context. Please try different encryption parameters.");
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
  // console.log("Secret key", secretKey);
  // console.log("Public key", publicKey);
  // console.log("Relin key", relinBase64Key);
  // console.log("Galois key", galoisBase64Key);
})();

class ClassificationComponent extends React.Component {
  themecolor = [100, 149, 237];

  constructor(props) {
    super(props);
    this.state = {
      prediction: -1,
      probabilities: [],
    };
  }

  classify() {
    console.log("Classifying given input");
    const self = this;
    const target = document.querySelector("#target-28x28");
    pica.resize(this.getDrawingCanvas(), target, { alpha: true }).then((result) => {
      console.log("Resizing to 28x28 finished.");
      let ctx = result.getContext("2d");
      let alphaChannel = ctx.getImageData(0, 0, 28, 28).data.filter((value, index) => index % 4 === 3); // alpha channel is the last of every 4-element-block.
      // TODO: rescale from 0..255 to 0..1
      console.log(alphaChannel);
      fetch("http://localhost:8000/api/classify/plain/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: Array.from(alphaChannel),
        }),
      }).then((response) => {
        response.json().then((data) =>
          self.setState({
            prediction: data.prediction,
            probabilities: data.probabilities,
          })
        );
      });
    });
  }

  clear() {
    let canvas = this.getDrawingCanvas();
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    this.drawGrid();
  }

  getDrawingCanvas() {
    return document.querySelector(".canvas-container canvas");
  }

  drawGrid() {
    console.log("Drawing grid.");
    const canvas = this.getDrawingCanvas();
    const ctx = canvas.getContext("2d");
    const cell = canvas.width / 28;  // width of one grid cell
    const gridSvg = `<svg width="${canvas.width}" height="${canvas.height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <pattern id="smallGrid" width="${cell}" height="${cell}" patternUnits="userSpaceOnUse">
                <path d="M ${cell} 0 L 0 0 0 ${cell}" fill="none" stroke="gray" stroke-width="0.5" />
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#smallGrid)" />
    </svg>`;
    var DOMURL = window.URL || window.webkitURL || window;
    var img = new Image();
    var svg = new Blob([gridSvg], { type: "image/svg+xml;charset=utf-8" });
    var url = DOMURL.createObjectURL(svg);

    img.onload = function () {
      ctx.drawImage(img, 0, 0);
      DOMURL.revokeObjectURL(url);
    };
    img.src = url;
  }

  componentDidMount() {
    this.drawGrid();
  }

  render() {
    const self = this;
    return (
      <Row>
        <Col m={6} s={12}>
          <ReactPainter
            width={280}
            height={280}
            initialColor={"cornflowerblue"}
            initialLineWidth={25}
            initialLineJoin={"miter"}
            lineCap={"round"}
            render={({ triggerSave, canvas, setColor }) => {
              return (
                <div>
                  <div className={"canvas-container center"}>{canvas}</div>
                  <Button onClick={self.classify.bind(self)}>Classify</Button>
                  <Button onClick={self.clear.bind(self)}>Clear</Button>
                </div>
              );
            }}
          />
        </Col>
        <Col m={6}>
          28x28 downscaled version:
          <canvas id="target-28x28" width={28} height={28}></canvas>
          <h3>
            Prediction: <b>{this.state.prediction}</b>
          </h3>
          <h5>Probabilities</h5>
          <ul className="probabilities">
            {this.state.probabilities.map((probability, index) => {
              let factor = 1 - probability / Math.max.apply(Math, this.state.probabilities);
              return (
                <li
                  key={index}
                  style={{
                    backgroundColor: `rgb(
                      ${factor * (255 - this.themecolor[0]) + this.themecolor[0]},
                      ${factor * (255 - this.themecolor[1]) + this.themecolor[1]},
                      ${factor * (255 - this.themecolor[2]) + this.themecolor[2]}
                  )`,
                  }}
                  title={probability}
                >
                  {index}
                </li>
              );
            })}
          </ul>
        </Col>
      </Row>
    );
  }
}

function App() {
  return (
    <div>
      <Navbar
        alignLinks="right"
        brand={<span style={{ paddingLeft: "10px" }}>FHE Classifier</span>}
        id="mobile-nav"
        menuIcon={<Icon>menu</Icon>}
      ></Navbar>
      <Container>
        <h3 className={"center"}>Classify your Secret Data</h3>
        <p>Using Fully Homomorphic Encryption, directly from within the browser.</p>
        <ClassificationComponent />
      </Container>
    </div>
  );
}

export default App;
