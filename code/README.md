Working directory for all your source code.

- Create secrets
    - `code/secrets/fheclassifier-password.txt` (e.g. 20 character password)
    - SSL Certificate and Private Key  
      `openssl req -x509 -newkey rsa:4096 -nodes -keyout code/secrets/fhe-classifier.key -out code/secrets/fhe-classifier.cert -days 3650 -subj "/C=AT/ST=Styria/L=Springfield/O=IAIK/CN=www.example.com"`
    - Secrets File `secrets.json`
- `docker-compose build`
- `docker-compose up`
