import React from "react";

export class DemoImageComponent extends React.Component {
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
