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
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 28, 28);
    let imgData = ctx.getImageData(0, 0, 28, 28);
    for (let i = 0; i < 28; i++)
      for (let j = 0; j < 28; j++) {
        const pixelIndex = (i * 28 + j) * 4,
          value = 255 - this.imageData[i][j] * 255;
        imgData.data[pixelIndex + 0] = value;
        imgData.data[pixelIndex + 1] = value;
        imgData.data[pixelIndex + 2] = value;
        imgData.data[pixelIndex + 3] = 255; // opaque
        if (value === 255) imgData.data[pixelIndex + 3] = 0; // transparent
      }
    ctx.putImageData(imgData, 0, 0);
  }

  render() {
    return (
      <canvas
        ref={this.canvasRef}
        width={28}
        height={28}
        onClick={this.props.onClick}
        style={{ cursor: "pointer" }}
      ></canvas>
    );
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
      testImagesAvailable: 0,
    };
    this.communicator = PlainCommunicator.instance();
    this.testImages = []; // outsource this from state because it is large
  }

  componentDidMount() {
    this.initGrid();
    this.fetchMoreTestImages();
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
    pica.resize(this.getDrawingCanvas(), target).then((result) => {
      let ctx = result.getContext("2d");
      // alpha channel is the last of every 4-element-block, so we have index % 4 == 3
      let alphaChannel = ctx.getImageData(0, 0, 28, 28).data.filter((value, index) => index % 4 === 3);
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
    const canvas = this.getDrawingCanvas();
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }

  initGrid() {
    const canvas = this.getDrawingCanvas();
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
    var img = document.querySelector(".background-grid");
    var svg = new Blob([gridSvg], { type: "image/svg+xml;charset=utf-8" });
    var url = DOMURL.createObjectURL(svg);
    img.src = url;
  }

  fetchMoreTestImages() {
    const alreadyThere = this.testImages.length;
    const indices = [...Array(80).keys()].map((i) => i + alreadyThere).join("-");
    console.log(indices, alreadyThere);
    fetch(`/api/testdata/?indices=${indices}`).then((response) => {
      response.json().then((data) => {
        this.testImages = this.testImages.concat(data);
        this.setState({
          testImagesAvailable: this.state.testImagesAvailable + 1,
        });
      });
    });
  }

  loadTestImage(event) {
    pica.resize(event.target, this.getDrawingCanvas());
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
                  <div className={"canvas-container center"}>
                    <img className="background-grid" alt="background grid" />
                    {canvas}
                  </div>
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
        {this.state.testImagesAvailable && (
          <Col s={12}>
            {this.testImages.map((img, index) => (
              <DemoImageComponent imageData={img} key={index} onClick={self.loadTestImage.bind(self)} />
            ))}
            <button
              type="button"
              className="btn-small btn-flat"
              style={{ marginTop: -20, marginLeft: 8 }}
              onClick={self.fetchMoreTestImages.bind(self)}
            >
              ... more
            </button>
          </Col>
        )}
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
