#!/bin/sh

if [ ! -f /etc/secrets/fhe-classifier.cert ] && [ -d /etc/demosecrets ]; then
  echo "Did not find secrets, so loaded demo secrets and wrote back"
  mkdir -p /etc/secrets/
  cp /etc/demosecrets/fhe-classifier.cert /etc/secrets/fhe-classifier.cert
  cp /etc/demosecrets/fhe-classifier.key /etc/secrets/fhe-classifier.key
fi

exec nginx -g "daemon off;"
