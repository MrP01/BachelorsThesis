# Part 1: C++ build process
# FROM conanio/gcc11 AS cpp-build
FROM alpine:edge AS base
WORKDIR /classifier
USER root
RUN apk add --no-cache libstdc++

FROM base AS cpp-build
RUN apk add --no-cache cmake git python3
RUN apk add --no-cache py3-pip
RUN python3 -m pip install --no-cache-dir conan

RUN apk add --no-cache make
RUN apk add --no-cache g++
# Maybe we can package SEAL into conan?
RUN git clone --depth 1 https://github.com/microsoft/SEAL.git /tmp/seal
RUN cd /tmp/seal \
  && cmake -S . -B build -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF \
  && cmake --build build -- -j 3 \
  && cmake --install build

RUN conan profile new default --detect --force \
  && conan profile update settings.compiler.libcxx=libstdc++11 default

COPY ./classifier/conanfile.txt /classifier/conanfile.txt
RUN mkdir /classifier/build/
RUN cd /classifier/build/ && conan install --build=backward-cpp --build=libdwarf --build=libelf ..

COPY ./classifier/ /classifier/
RUN cd /classifier/build/ && cmake .. && cmake --build . -- -j 3

# Part 2: Train Neural Network
FROM python:3.10 AS trainer
WORKDIR /training

RUN pip install --upgrade pip poetry==1.2.0b1
COPY ./pyproject.toml /training/pyproject.toml
COPY ./poetry.lock /training/poetry.lock
RUN poetry config virtualenvs.create false && poetry install --no-interaction

RUN mkdir -p /classifier/data/mnist/ /classifier/data/models/simple/
COPY ./tasks.py /tasks.py
RUN --mount=type=cache,target=/root/.keras/datasets/ inv fetch-training-data

COPY ./training/ /training/
RUN --mount=type=cache,target=/root/.keras/datasets/ python /training/network.py

# Part 3: Tiny image only containing the binary, the model and training+test data
FROM base
COPY --from=cpp-build /root/.conan/data/ /root/.conan/data/
COPY --from=cpp-build /classifier/build/bin/classifier /classifier/classifier
COPY --from=trainer /classifier/data/ /classifier/data/
CMD ["/classifier/classifier"]
