#!/usr/bin/env sh

./wait-for-it.sh db:5432
./manage.py migrate
