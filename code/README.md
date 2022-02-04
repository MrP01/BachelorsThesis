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

```bash
cd /path/to/the/repo
cd code/

# Training:
cd training/
pip install -r requirements.txt
python3 network.py
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
./build/bin/classifier  # start the server
cd ../

# Frontend:
cd frontend
yarn install
yarn start
cd ../
```
