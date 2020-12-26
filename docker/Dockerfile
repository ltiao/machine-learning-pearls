FROM tensorflow/tensorflow:2.3.0-gpu-jupyter
MAINTAINER Louis Tiao <louistiao@gmail.com>

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get update && apt-get install -y --no-install-recommends \
        git-core \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-xetex \
        cm-super \
        dvipng \
        pandoc \
        msttcorefonts \
        poppler-utils \
        imagemagick \
        ffmpeg \
        graphviz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ~/.cache/matplotlib

# TODO: copy to temporary dir rather than unknown current dir
COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==20.3.3 keyrings.alt==3.4.0 && \
    python -m pip install --no-cache-dir -r requirements_dev.txt && \
	python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --upgrade https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.52-cp36-none-manylinux2010_x86_64.whl && \
    python -m pip install --upgrade jax

# Install GPFlow (taking care of its dependencies ourselves)
RUN python -m pip install --no-cache-dir multipledispatch tabulate dataclasses && \
    python -m pip install --no-cache-dir --no-deps gpflow==2.0.1
RUN python -m pip install --upgrade GPy==1.9.9 gpyopt==1.2.6

RUN mkdir -p /usr/src/app

COPY . /usr/src/app
WORKDIR /usr/src/app
RUN python -m pip install --no-deps -e .

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
