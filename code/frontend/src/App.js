import React from "react";
import "@materializecss/materialize";
import "@materializecss/materialize/dist/css/materialize.css";
import { Navbar, Container } from "react-materialize";
import "./App.css";

import { ClassificationComponent } from "./ClassificationComponent";

function App() {
  return (
    <div>
      <Navbar
        alignLinks="right"
        brand={<span style={{ paddingLeft: "10px", width: "220px" }}>FHE Classifier</span>}
        id="mobile-nav"
        menuIcon={<span />}
        sidenav={null}
        style={{ backgroundColor: "cornflowerblue" }}
      ></Navbar>
      <Container>
        <h3 className={"center"}>Classify your Secret Data</h3>
        <p className={"center"}>
          Using state-of-the-art Fully Homomorphic Encryption, directly from within the browser, based on Web Assembly.
        </p>
        <ClassificationComponent />
      </Container>
    </div>
  );
}

export default App;
