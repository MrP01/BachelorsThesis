Working directory for all your source code.

## For running with Docker (recommended):

- Create secrets
  - `code/secrets/fheclassifier-password.txt` (e.g. 20 character password)
  - SSL Certificate and Private Key
    `openssl req -x509 -newkey rsa:4096 -nodes -keyout code/secrets/fhe-classifier.key -out code/secrets/fhe-classifier.cert -days 3650 -subj "/C=AT/ST=Styria/L=Springfield/O=IAIK/CN=www.example.com"`
  - Secrets File `secrets.json`
- `docker-compose build`
- `docker-compose up`

## For local development

For Python, install [poetry](https://python-poetry.org/).

The best way to do so is using [pipx](https://pypa.github.io/pipx/)
(which performs a global user-installation).

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
poetry shell  # creates and activates a new virtual environment
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
