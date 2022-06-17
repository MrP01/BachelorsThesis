# Welcome to the code

The project is structured into the

- `training/` component that trains and stores the neural network (Python),
- `classifier/` component running as the backend API server, using SEAL (C++),
- `frontend/` component which displays a small web UI (JavaScript).

## For running with Docker (recommended):

The updated setup is completely ready-to-go as-is and can be pulled from the official Docker registry.

To do so, simply create a file named `docker-compose.yml` somewhere with the following contents:

```yaml
version: "3.7"
services:
  classifier:
    image: mrp001/sealed-mnist-classifier

  frontend:
    image: mrp001/sealed-mnist-frontend
    ports:
      - 80:80
      - 443:443
    volumes:
      - ./secrets/:/etc/secrets/
```

Docker-Compose will do the rest for you:

```bash
docker-compose up
```

Alternatively, without creating the `docker-compose.yml`:

```bash
docker run --name classifier --detach mrp001/sealed-mnist-classifier
docker run -p 80:80 -p 443:443 --link classifier mrp001/sealed-mnist-frontend
```

## For local development

For Python, install [poetry](https://python-poetry.org/).

The best way to do so is using [pipx](https://pypa.github.io/pipx/)
(which performs a user-wide, isolated installation).

`pipx install git+https://github.com/python-poetry/poetry.git`

Or use the other installation method described [here](https://python-poetry.org/docs/#installation).

Also, because SEAL is not yet packaged with Conan, this is
[the official way](https://github.com/microsoft/SEAL#building-microsoft-seal-manually) to install it:

```bash
git clone git@github.com:microsoft/SEAL.git

cd SEAL/
cmake -S . -B build -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF
cmake --build build
sudo cmake --install build
```

This will add the headers to your system path and SEAL will be available when compiling.

To run the present project:

```bash
git clone git@extgit.iaik.tugraz.at:crypto_students/2020_bachelor_waldert.git

cd 2020_bachelor_waldert/
cd code/
poetry shell  # creates and activates a new virtual environment (run this every time to activate it)
poetry install  # installs dependencies from pyproject.toml

# Training:
cd training/
python network.py  # trains the model and stores it
cd ../

# Classifier:
cd classifier/
pip install conan
conan profile new default --detect
conan profile update settings.compiler.libcxx=libstdc++11 default
mkdir build
cd build/
conan install ..
cmake ..
cmake --build . -- -j 3
cd ../
./build/bin/evaluate  # either evaluate the network on test data (plain and encrypted), or:
./build/bin/classifier  # start the backend server
cd ../

# Frontend:
cd frontend/
yarn install
yarn start  # runs the frontend and opens a browser tab
cd ../
```
