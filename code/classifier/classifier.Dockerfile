# syntax = docker/dockerfile:experimental
FROM conanio/gcc10
WORKDIR /app
USER root

# Maybe we can package SEAL into conan?
RUN git clone --depth 1 https://github.com/microsoft/SEAL.git /tmp/seal \
    && cd /tmp/seal \
    && cmake -S . -B build \
    && cmake --build build -- -j 3 \
    && cmake --install build

RUN conan profile new default --detect --force \
    && conan profile update settings.compiler.libcxx=libstdc++11 default

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
COPY tasks.py /app/tasks.py
# RUN --mount=type=cache,target=/root/.local/share/MNIST/ inv fetch-training-data

COPY conanfile.txt /app/conanfile.txt
RUN mkdir /app/cmake-build-debug/
RUN cd /app/cmake-build-debug/ && conan install ..

COPY . /app
RUN cd /app/cmake-build-debug/ && cmake .. && cmake --build . -- -j 3

CMD /app/cmake-build-debug/bin/classifier
