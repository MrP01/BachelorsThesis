FROM conanio/gcc10
WORKDIR /app

COPY . /app
RUN mkdir /app/cmake-build-debug/
RUN cd /app/cmake-build-debug/ && conan install ..

RUN git clone https://github.com/microsoft/SEAL.git /tmp/seal && cd /tmp/seal
RUN cmake -S . -B build
RUN cmake --build build
RUN sudo cmake --install build

RUN cd /app
RUN cmake --build /app/cmake-build-debug --target classifier -- -j 3

CMD /app/cmake-build-debug/bin/classifier
