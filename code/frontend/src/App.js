import 'materialize-css';
import 'materialize-css/dist/css/materialize.css';
import {Button, Col, Icon, Navbar, Row} from 'react-materialize';
import "./App.css";

function App() {
  return (
    <div>
      <Navbar
        alignLinks="right"
        brand={<a href="#" style={{"padding-left": "10px"}}>FHE Classifier</a>}
        id="mobile-nav"
        menuIcon={<Icon>menu</Icon>}
      >
      </Navbar>
      <Row>
        <Col s={12}>
          <h3 className={"center"}>Classify your Secret Data</h3>
          <p>Using Fully Homomorphic Encryption, directly from within the browser.</p>
          <Button>Upload</Button>
        </Col>
      </Row>
    </div>
  );
}

export default App;
