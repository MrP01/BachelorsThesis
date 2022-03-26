import React from "react";
import Pica from "pica";
import "@materializecss/materialize";
import "@materializecss/materialize/dist/css/materialize.css";
import { Button, Col, Navbar, Row, Container, Switch, ProgressBar } from "react-materialize";
import { ReactPainter } from "react-painter";
import { PlainCommunicator, SEALCommunicator } from "./communicators";
import "./App.css";

const pica = new Pica();

class DemoImageComponent extends React.Component {
  constructor(props) {
    super(props);
    this.imageData = props.imageData;
    this.canvasRef = React.createRef();
  }

  componentDidMount() {
    let ctx = this.canvasRef.current.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 28, 28);
    let imgData = ctx.getImageData(0, 0, 28, 28);
    for (let i = 0; i < 28; i++)
      for (let j = 0; j < 28; j++) {
        imgData.data[(i * 28 + j) * 4 + 0] = this.imageData[i][j];
        imgData.data[(i * 28 + j) * 4 + 1] = this.imageData[i][j];
        imgData.data[(i * 28 + j) * 4 + 2] = this.imageData[i][j];
      }
    ctx.putImageData(imgData, 0, 0);
  }

  render() {
    return <canvas ref={this.canvasRef} width={28} height={28}></canvas>;
  }
}

class ClassificationComponent extends React.Component {
  themecolor = [100, 149, 237]; // cornflowerblue

  constructor(props) {
    super(props);
    this.state = {
      prediction: "...",
      probabilities: [...Array(10).keys()].map((x) => x / 10),
      calculating: false,
      testImagesAvailable: false,
    };
    this.communicator = PlainCommunicator.instance();
    this.testImages = [];
    fetch("/api/testdata/?indices=-3-4-5").then((response) => {
      response.json().then((data) => {
        this.testImages = data;
        this.setState({
          testImagesAvailable: true,
        });
      });
    });
  }

  componentDidMount() {
    this.drawGrid();
  }

  componentWillUnmount() {
    // this.communicator.delete();
    // delete this.communicator;
    // console.log("Cleaned up.");
  }

  classify() {
    console.log("Classifying given input");
    const self = this;
    const target = document.querySelector("#target-28x28");
    self.setState({ calculating: true, prediction: "..." });
    pica.resize(this.getDrawingCanvas(), target, { alpha: true }).then((result) => {
      console.log("Resizing to 28x28 finished.");
      let ctx = result.getContext("2d");
      let alphaChannel = ctx.getImageData(0, 0, 28, 28).data.filter((value, index) => index % 4 === 3); // alpha channel is the last of every 4-element-block.
      // TODO: rescale from 0..255 to 0..1
      alphaChannel = alphaChannel.map((x) => (x > 127 ? 1 : 0));
      console.log(alphaChannel);
      // give the browser one extra cycle for rendering
      setTimeout(
        () =>
          this.communicator
            .classify(alphaChannel)
            .then((data) =>
              self.setState({
                prediction: data.prediction,
                probabilities: data.probabilities,
                calculating: false,
              })
            )
            .catch((err) => self.setState({ calculating: false })),
        0
      );
    });
  }

  getDrawingCanvas() {
    return document.querySelector(".canvas-container canvas");
  }

  clear() {
    let canvas = this.getDrawingCanvas();
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    this.drawGrid();
  }

  drawGrid() {
    console.log("Drawing grid.");
    const canvas = this.getDrawingCanvas();
    const ctx = canvas.getContext("2d");
    const cell = canvas.width / 28; // width of one grid cell
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

  setCommunicator(event) {
    this.communicator = (event.target.checked ? SEALCommunicator : PlainCommunicator).instance();
  }

  render() {
    const self = this;
    return (
      <Row>
        <ProgressBar className={self.state.calculating ? "" : "transparent"} />
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
                  <div className="command-bar">
                    <Button onClick={self.clear.bind(self)}>Clear</Button>
                    <Button onClick={self.classify.bind(self)} disabled={self.state.calculating}>
                      Classify
                    </Button>
                  </div>
                </div>
              );
            }}
          />
          <p className="grey-text">Each grid cell represents one pixel in the 28x28 image.</p>
        </Col>
        <Col m={6}>
          <div className="card-panel z-depth-1">
            <div className="row valign-wrapper">
              <div className="col s2">
                <canvas id="target-28x28" className="center-block" width={28} height={28}></canvas>
              </div>
              <div className="col s10">
                <span className="black-text">The 28x28 downscaled version will be classified using the</span>
                <Switch
                  offLabel="PlainCommunicator"
                  onLabel="SEALCommunicator"
                  onChange={self.setCommunicator.bind(self)}
                />
              </div>
            </div>
          </div>
          <h3 className="center-align">
            Prediction: <b>{this.state.prediction}</b>
          </h3>
          <h6>Probabilities</h6>
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
        <Col>
          {this.testImages.map((img, index) => (
            <DemoImageComponent imageData={img} key={index} />
          ))}
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
        menuIcon={<span />}
        style={{ backgroundColor: "cornflowerblue" }}
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
