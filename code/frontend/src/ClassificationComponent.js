import React from "react";
import Pica from "pica";
import { Button, Col, Row, Switch, ProgressBar } from "react-materialize";
import { ReactPainter } from "react-painter";

import { PlainCommunicator, SEALCommunicator } from "./communicators";
import { DemoImageComponent } from "./DemoImageComponent";
import { ProbabilityDisplay } from "./ProbabilityDisplay";

const pica = new Pica();

export class ClassificationComponent extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      prediction: "...",
      probabilities: [...Array(10).keys()].map((x) => x / 10),
      calculating: false,
      testImagesAvailable: 0,
    };
    this.communicator = null;
    const self = this;
    PlainCommunicator.instance().then((comm) => (self.communicator = comm));
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
    this.setState({ calculating: true });
    (event.target.checked ? SEALCommunicator : PlainCommunicator).instance().then((comm) => {
      this.communicator = comm;
      this.setState({ calculating: false });
    });
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
        <Col m={6} s={12}>
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
                This will take up browser resources for a few seconds.
              </div>
            </div>
          </div>
          <h3 className="center-align">
            Prediction: <b>{this.state.prediction}</b>
          </h3>
          <h6>Probabilities</h6>
          <ProbabilityDisplay probabilities={this.state.probabilities} />
        </Col>
        {this.state.testImagesAvailable && (
          <Col s={12}>
            <p style={{ marginTop: 0, marginBottom: 4 }} className={"center"}>
              By clicking on one of the following test images, you can load it to the canvas directly:
            </p>
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
