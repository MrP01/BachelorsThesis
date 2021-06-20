import 'materialize-css';
import 'materialize-css/dist/css/materialize.css';
import { Button, Col, Icon, Navbar, Row } from 'react-materialize';
import { ReactPainter } from 'react-painter';
import "./App.css";
import SEAL from 'node-seal/throws_wasm_node_umd'

const seal = await SEAL();

function classify(blob) {
  console.log("Classifying given input", blob);
}

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
