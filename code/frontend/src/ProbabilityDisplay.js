import React from "react";

export class ProbabilityDisplay extends React.Component {
  themecolor = [100, 149, 237]; // cornflowerblue

  render() {
    return (
      <ul className="probabilities">
        {this.props.probabilities.map((probability, index) => {
          let factor = 1 - probability / Math.max.apply(Math, this.props.probabilities);
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
    );
  }
}
