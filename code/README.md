# Welcome to the code

The project is structured into the

- `network/` component that trains and stores the neural network (Python),
- `classifier/` component running as the backend, using SEAL (C++),
- `frontend/` component which displays a small web UI (JavaScript).

## For running with Docker (recommended):

```bash
cd code/
source .env  # which contains the path to the secrets folder
inv generate-secrets  # creates secrets: SSL Certificate and Private Key for nginx
docker-compose build  # builds and compiles everything, will take some time
docker-compose up
```

## For local development

For Python, install [poetry](https://python-poetry.org/).

The best way to do so is using [pipx](https://pypa.github.io/pipx/)
(which performs a global, isolated user-installation).

`pipx install poetry`

Or use the other installation method described [here](https://python-poetry.org/docs/#installation).

Also, because SEAL is not yet packaged with Conan, this is
[the official way](https://github.com/microsoft/SEAL#building-microsoft-seal-manually) to install it:

```bash
git clone git@github.com:microsoft/SEAL.git

cd SEAL/
cmake -S . -B build
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
