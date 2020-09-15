# source https://github.com/mingfang/docker-geekbench
FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-setuptools \
    python3-pip \
    pkg-config \
    libfreetype6-dev \
    libpng12-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    python3-h5py \
    libjpeg8-dev \
    python3-tk \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxtst6

RUN pip3 install --upgrade \
    numpy \
    matplotlib \
    scipy \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp35-cp35m-linux_x86_64.whl \
    keras \
    tqdm \
    pillow \
    scikit-learn \
    opencv-python \
    scikit-image


ENTRYPOINT ["python3", "-u"]
WORKDIR /opt/
