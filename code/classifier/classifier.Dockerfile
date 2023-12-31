# Part 1: C++ build process
FROM alpine:edge AS base
WORKDIR /classifier
USER root
RUN apk add --no-cache libstdc++ gmp

FROM base AS cpp-build
RUN apk add --no-cache cmake git python3 py3-pip make g++ perl
RUN python3 -m pip install --no-cache-dir conan

RUN apk add --no-cache gmp-dev
RUN git clone --depth 1 https://github.com/libntl/ntl.git /tmp/ntl
RUN cd /tmp/ntl/src \
  && ./configure SHARED=on NTL_GMP_LIP=on NTL_THREADS=on NTL_THREAD_BOOST=on NTL_RANDOM_AES256CTR=on \
  && make -j4 && make install && ldconfig /

# Maybe we can package SEAL into conan?
RUN git clone -b v4.0.0 --depth 1 https://github.com/microsoft/SEAL.git /tmp/seal
RUN cd /tmp/seal \
  && cmake -S . -B build -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF \
  && cmake --build build -- -j3 \
  && cmake --install build

RUN conan profile new default --detect --force \
  && conan profile update settings.compiler.libcxx=libstdc++11 default
COPY ./classifier/conanfile.txt /classifier/conanfile.txt
RUN mkdir /classifier/build/
RUN cd /classifier/build/ && conan install --build=lodepng ..

COPY ./classifier/ /classifier/
RUN cd /classifier/build/ && cmake .. && CMAKE_BUILD_TYPE=RelWithDebInfo cmake --build . -- -j3

# Part 2: Train Neural Network
FROM python:3.11 AS trainer
WORKDIR /
RUN pip install --upgrade pip poetry
COPY ./pyproject.toml /pyproject.toml
COPY ./poetry.lock /poetry.lock
RUN poetry config virtualenvs.create false && poetry install --no-interaction --only=main
RUN mkdir -p /classifier/data/mnist/ /classifier/data/models/simple/
COPY ./tasks.py /tasks.py
RUN --mount=type=cache,target=/root/.keras/datasets/ python -m invoke fetch-training-data

COPY ./network.py /network.py
RUN --mount=type=cache,target=/root/.keras/datasets/ python -m invoke train

# Part 3: Tiny image only containing the binary, the model and test data
FROM base
COPY --from=cpp-build /usr/local/lib/libntl.so.44 /usr/local/lib/libntl.so.44
COPY --from=cpp-build /classifier/build/bin/classifier /classifier/classifier
COPY --from=cpp-build /classifier/build/bin/evaluate /classifier/evaluate
COPY --from=trainer /classifier/data/models/ /classifier/data/models/
COPY --from=trainer /classifier/data/mnist/x-test.npy /classifier/data/mnist/x-test.npy
COPY --from=trainer /classifier/data/mnist/y-test.npy /classifier/data/mnist/y-test.npy
RUN mkdir /classifier/data/ciphertext-visualisation/
CMD ["/classifier/classifier"]
