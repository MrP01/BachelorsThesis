#!/usr/bin/env sh

wait-for-it --service db:5432
./manage.py migrate
