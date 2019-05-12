FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip

RUN pip3 install \
    pylint \
    jupyter \
    numpy \
    scipy \
    matplotlib \
    pandas \
    seaborn \
    statsmodels \
    scikit-learn \
    tensorflow

WORKDIR /workspace