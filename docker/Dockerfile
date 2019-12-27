FROM tensorflow/tensorflow:1.15.0-py3-jupyter
MAINTAINER Louis Tiao <louistiao@gmail.com>

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get update && apt-get install -y --no-install-recommends \
        # texlive-full \
        # msttcorefonts \
        git-core \
        graphviz \
        ffmpeg \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ~/.cache/matplotlib

COPY requirements_dev.txt .
RUN python -m pip install --no-cache-dir -r requirements_dev.txt

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app
RUN python -m pip install -e .
