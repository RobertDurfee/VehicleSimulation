FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN useradd -ms /bin/bash python
USER python

WORKDIR /home/python/work